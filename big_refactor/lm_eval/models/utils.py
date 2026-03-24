import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_rank
import os
try:
    from transformers import LlamaMoEV7ForCausalLM
    from transformers import LlamaAttention
except Exception:
    pass
from transformers import AutoConfig, BitsAndBytesConfig
import json
from peft import PeftModel


def rank0_print(*args, **kwargs):
    try:
        if get_rank() == 0:
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)


# class RouterV7(nn.Module):
#     """
#     基于multi-head attention构建
#     """
#     def __init__(self, config, expert_nums):
#         super().__init__()
#         self.config = config
#         self.expert_nums = expert_nums
#         self.multihead_attn = nn.MultiheadAttention(
#             embed_dim=self.config.hidden_size, 
#             num_heads=8, 
#             dropout=0.05, 
#             kdim=self.config.hidden_size,
#             vdim=self.config.hidden_size,
#             batch_first=True
#         )
#         self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(self.config.hidden_size // 2, self.expert_nums)
    
#     def forward(self, inputs_embeds, attn_mask):
#         x, _ = self.multihead_attn(
#             query=inputs_embeds,
#             key=inputs_embeds,
#             value=inputs_embeds,
#             key_padding_mask=attn_mask,
#         )
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
        
#         return torch.softmax(x, dim=-1)


class RouterV7(nn.Module):
    """
    基于multi-head attention构建
    """
    def __init__(self, config, expert_nums):
        super().__init__()
        self.config = config
        self.expert_nums = expert_nums
        self.multihead_attn = LlamaAttention(self.config)
        self.fc1 = nn.Linear(self.config.hidden_size, 128)
        self.activate = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, self.expert_nums)
    
    def forward(self, inputs_embeds, attn_mask):
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        x, _, _ = self.multihead_attn(
            hidden_states=inputs_embeds,
            attention_mask=attn_mask,
        )
        # residual connection
        x = x + inputs_embeds

        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        # residual connection
        
        return x


def create_MoEV7model(model_name_or_path, adapter_map_path):
    def set_requires_grad(model):
        # 设置参数的requires_grad属性
        for name, p in model.named_parameters():
            if "router" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    n_gpus = torch.cuda.device_count()
    max_memory = f'80000MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    rank0_print("moe_version: 7")

    adapter_map = json.load(open(adapter_map_path))
    assert isinstance(adapter_map, dict)

    base_model_path = model_name_or_path
    print(f"loading base model from {base_model_path}")
    config = AutoConfig.from_pretrained(base_model_path)
    model = LlamaMoEV7ForCausalLM.from_pretrained(
        base_model_path, 
        device_map=device_map, 
        config=config, 
        expert_nums=len(adapter_map),
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    for idx, (name, path) in enumerate(adapter_map.items()):
        if idx == 0:
            peft_model = PeftModel.from_pretrained(model, path, name)
            rank0_print("loading adapter `{}` from `{}`".format(name, path))
        else:
            peft_model.load_adapter(path, name)
            rank0_print("loading adapter `{}` from `{}`".format(name, path))
    else:
        rank0_print("adding router".center(200, ">"))
        peft_model.model.model.router = RouterV7(model.config, len(adapter_map)).to(peft_model.model.model.embed_tokens.weight.device)
        for p in peft_model.model.model.router.parameters():
            p.data = p.data.to(torch.bfloat16)
        set_requires_grad(peft_model)
        return peft_model