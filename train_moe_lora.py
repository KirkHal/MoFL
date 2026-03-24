import copy
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Dict, Sequence
import logging
import argparse
import bitsandbytes as bnb
import pandas as pd
from datasets import load_dataset, Dataset
from conversation import get_default_conv_template, SeparatorStyle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.distributed import get_rank

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import LabelSmoother

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
)
from peft.tuners.lora import LoraLayer
from peft import (
    PeftModel,
    MoELoraModel,      # MoLoRA
    MoELoraModelV2,    # Convergent MoE
    MoELoraModelV3,    # Divergent MoE
    MoELoraModelV4,    # MoFL-1
    MoELoraModelV5,    # MoFL-4
)
from utils.args import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from utils.utils import print_trainable_parameters, get_last_checkpoint


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100




def rank0_print(*args, **kwargs):
    try:
        if get_rank() == 0:
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)



def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_len: int
    dataset_format: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.dataset_format == "icbu-tars-multi":
            sources = [f"{example['input']}" for example in instances]  # conversations
            input_ids = self.tokenizer(
                sources,
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_max_len,
                truncation=True,
                add_special_tokens=False,
            ).input_ids
            targets = input_ids.clone()
            conv = get_default_conv_template(model_id).copy()
            task_ids = [example["task_id"] for example in instances]

            if conv.sep_style == SeparatorStyle.LLAMA2_CHAT:

                sep_ua_message = " </s><s>"  # split into user-assistant rounds  [29871, 2, 1]
                sep_between_ua = "[/INST] "  # split into user-assistant message pairs  [518, 29914, 25580, 29962, 29871]

                for conversation, target in zip(sources, targets):
                    total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
                    rounds = conversation.split(sep_ua_message)
                    cur_len = 0
                    for i, rou in enumerate(rounds):
                        if rou == "":
                            break
                        parts = rou.split(sep_between_ua)
                        rou_w_sep = rou + sep_ua_message
                        if len(parts) != 2:  # no "[/INST] " in between, its a "system_prompt + 1st_user_message" section for llama-2, which should be masked
                            rank0_print(
                                "no [/INST]  in between, its a system_prompt + 1st_user_message section for llama-2, which should be masked")
                            rank0_print("conversation:", conversation)
                            rank0_print("rou_w_sep:", rou_w_sep)
                            mask_len = len(self.tokenizer(rou_w_sep,
                                                          add_special_tokens=False).input_ids) - 1  # for [BOS] & [EOS] positions
                            target[max(cur_len - 2, 0): cur_len + mask_len] = IGNORE_INDEX
                        else:
                            use_msg, asst_msg = parts
                            use_msg_w_sep = use_msg + sep_between_ua
                            mask_len = len(self.tokenizer(use_msg_w_sep, add_special_tokens=False).input_ids) - 1
                            target[max(cur_len - 2, 0): cur_len + mask_len] = IGNORE_INDEX
                            if i < len(rounds) - 1:
                                round_len = len(self.tokenizer(rou_w_sep, add_special_tokens=False).input_ids)
                            else:  # last one round
                                round_len = len(self.tokenizer(rou, add_special_tokens=False).input_ids)

                            cur_len += round_len

                    target[cur_len - 2:] = IGNORE_INDEX

                # Apply padding
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                labels = pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
                data_dict = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
                }

                return data_dict

            else:

                assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO, "This branch is conv_llama_2 exclusive, you should add another branch."
                sep_ua_message = "</s>"  # split into user-assistant rounds  [29871, 2, 1]
                sep_between_ua = " ASSISTANT: "  # split into user-assistant message pairs  [518, 29914, 25580, 29962, 29871]

                for conversation, target in zip(sources, targets):
                    # rank0_print("*" * 80)
                    # rank0_print(conversation)
                    # rank0_print("+" * 80)

                    total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
                    rounds = conversation.split(sep_ua_message)
                    cur_len = 0
                    for i, rou in enumerate(rounds):
                        if rou == "":
                            break
                        parts = rou.split(sep_between_ua)
                        rou_w_sep = rou + sep_ua_message
                        if len(parts) != 2:  # no "[/INST] " in between, its a "system_prompt + 1st_user_message" section for llama-2, which should be masked
                            rank0_print(
                                "no [/INST]  in between, its a system_prompt + 1st_user_message section for llama-2, which should be masked")
                            rank0_print("conversation:", conversation)
                            rank0_print("rou_w_sep:", rou_w_sep)
                            mask_len = len(self.tokenizer(rou_w_sep,
                                                          add_special_tokens=False).input_ids) - 1  # for [BOS] & [EOS] positions
                            target[max(cur_len - 2, 0): cur_len + mask_len] = IGNORE_INDEX
                        else:
                            use_msg, asst_msg = parts
                            use_msg_w_sep = use_msg + sep_between_ua
                            mask_len = len(self.tokenizer(use_msg_w_sep, add_special_tokens=False).input_ids) - 1
                            target[max(cur_len - 2, 0): cur_len + mask_len] = IGNORE_INDEX
                            if i < len(rounds) - 1:
                                round_len = len(self.tokenizer(rou_w_sep, add_special_tokens=False).input_ids)
                            else:  # last one round
                                round_len = len(self.tokenizer(rou, add_special_tokens=False).input_ids)

                            cur_len += round_len

                    target[cur_len - 2:] = IGNORE_INDEX

                # Apply padding
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                labels = pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
                task_ids = torch.LongTensor(task_ids) if None not in task_ids else torch.LongTensor([1] * len(targets))

                data_dict = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
                }
                return data_dict

        else:
            # Extract elements
            sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
            targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
            # Tokenize
            tokenized_sources_with_prompt = self.tokenizer(
                sources,
                max_length=self.model_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = self.tokenizer(
                targets,
                max_length=self.model_max_len,
                truncation=True,
                add_special_tokens=False,
            )

            # Build the input and labels for causal LM
            input_ids = []
            labels = []
            for tokenized_source, tokenized_target in zip(
                    tokenized_sources_with_prompt['input_ids'],
                    tokenized_targets['input_ids']
            ):
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                )
            # Apply padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            data_dict = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            }

            return data_dict



def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out



def local_dataset(dataset_name):
    if dataset_name.endswith(('.json', '.jsonl')):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.01)
    return split_dataset



def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except Exception:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


    def icbu_tars_data_multi_preprocess(source):
        conv = get_default_conv_template(model_id).copy()
        roles = {
            "human": conv.roles[0],
            "gpt": conv.roles[1],
            "system": "system",
            "user": conv.roles[0],
            "bard": conv.roles[1],
            "bing": conv.roles[1],
            "chatgpt": conv.roles[1],
        }

        system_type = source.get("system", None)
        task_id = source.get("task_id", None)
        asst_type = None
        source = source["conversations"]

        if len(source) > 0 and roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                conv.messages = []  ### give up this conversation
                rank0_print("skip conversation: ", sentence["value"][:50])
                break

            conv.append_message(role, sentence["value"])
        instruction = conv.get_prompt(system_type, asst_type)
        if instruction is None:
            rank0_print("instruction is None", source)
            instruction = ""

        return {
            'input': instruction,
            'output': "",
            "task_id": task_id,
        }

    def format_dataset(dataset, dataset_format):
        if dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'icbu-tars-multi':
            if isinstance(dataset, dict):
                for k in dataset:
                    dataset[k] = dataset[k].map(lambda x: icbu_tars_data_multi_preprocess(x))
            else:
                dataset = dataset.map(lambda x: icbu_tars_data_multi_preprocess(x))
        elif dataset_format == 'input-output':
            # leave as is
            pass
        return dataset

    # Load dataset
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        rank0_print("+" * 50)
        rank0_print("train_dataset:", train_dataset)
        rank0_print("+" * 50)
        # train_dataset = dataset
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    # Remove any training data that exceeds the max length
    def _get_data_length(item):
        prompt = f"{tokenizer.bos_token}{item['input']}{item['output']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                max_length=args.model_max_len + 1,
                truncation=True,
                add_special_tokens=False
            ).input_ids
        )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_len=args.model_max_len,
        dataset_format=args.dataset_format
    )
    rank0_print("+" * 50)
    rank0_print("eval_dataset:", eval_dataset)
    rank0_print("+" * 50)
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )



class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        rank0_print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)



def set_requires_grad(args, model):
    for name, p in model.named_parameters():
        if "router" in name or "lora" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False



def get_moe_model(args, checkpoint_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    PREPARED_MOE_VERSION = {
        1: MoELoraModel,
        2: MoELoraModelV2,
        3: MoELoraModelV3,
        4: MoELoraModelV4,
        5: MoELoraModelV5,
    }

    if args.full_finetune:
        assert args.bits in [16, 32]

    rank0_print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "load_in_4bit": args.bits == 4,
        "load_in_8bit": args.bits == 8,
        "device_map": device_map,
        "max_memory": max_memory,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        "torch_dtype": (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
        "use_auth_token": args.use_auth_token
    }
    if args.mpt:
        model_kwargs["attn_config"] = {"attn_impl": "triton"}
    # EDITED
    if "flan-t5" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            rank0_print('=' * 80)
            rank0_print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            rank0_print('=' * 80)

    setattr(model, 'model_parallel', False)
    setattr(model, 'is_parallelizable', False)

    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            rank0_print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
            # return model
            raise NotImplementedError
        else:
            rank0_print("Adding LoRA modules...")
            if args.lora_modules == "all":
                modules = find_all_linear_names(args, model)
            else:
                modules = ["gate_proj", "down_proj", "up_proj"]
            # all_modules : ['gate_proj', 'down_proj', 'up_proj', 'k_proj', 'v_proj', 'q_proj', 'o_proj']

            if "flan-t5" in args.model_name_or_path:
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=['wi_1', 'wi_0'],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM",
                )
            else:
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            model = PREPARED_MOE_VERSION.get(args.moe_version)(model, config, args.expert_nums)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    set_requires_grad(args, model)
    return model



def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    global model_id
    model_id = args.model_id
    assert model_id is not None
    rank0_print("Selected model_id: ", model_id)

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "padding_side": "right",
        "use_fast": False,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.mpt:
        tokenizer_kwargs["padding_side"] = "left"
        tokenizer_kwargs.pop("use_fast")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    try:
        tokenizer.pad_token_id = tokenizer.eod_id
    except Exception:
        pass

    # Load data
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        rank0_print("Detected that training was already completed!")

    # Load model
    model = get_moe_model(args, checkpoint_dir)
    try:
        torch.distributed.barrier()
    except Exception:
        pass
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    rank0_print("Moe model loaded.")
    # breakpoint()


    set_seed(args.seed)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    # Verifying datatypes
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        rank0_print(k, v, v / total)


    # Start Training
    if args.do_train:
        logger.info("****** START TRAINING ******")
        trainer.train()
        print_trainable_parameters(args, model)

        # save state dict
        os.makedirs(f"{args.output_dir}", exist_ok=True)

        torch.save(model.state_dict(), f"{args.output_dir}/quant_state_dict.pth")
        trainer.save_state()



if __name__ == "__main__":
    train()
