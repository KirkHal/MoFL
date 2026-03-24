#!/bin/bash


foundation=llama-2-7b
dataset_path= #CUSTOM_DATASET_PATH
model_name_or_path= #CUSTOM_MODEL_PATH
dataset=puffin
conv_temp=llm2
train_epoch=3
lr=3e-4
lr_scheduler=constant_with_warmup
num_gpu=2
moe_version=5
lora_modules=all
expert_num=2
lora_r=4
custom= #CUSTOM_TAG
model_id=${foundation}_${dataset}_${custom}_${conv_temp}_MoEV${moe_version}_exp${expert_num}
out_dir=trained_models/${model_id}


unset WANDB_API_KEY
unset WANDB_PROJECT


CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=21005 train_moe_lora.py \
	--num_train_epochs ${train_epoch} \
	--logging_steps 1 \
	--data_seed 10818 \
	--evaluation_strategy steps \
	--eval_dataset_size 400 \
	--max_new_tokens 3072 \
	--model_name_or_path ${model_name_or_path} \
	--output_dir ${out_dir} \
	--dataloader_num_workers 3 \
	--logging_strategy steps \
	--remove_unused_columns False \
	--do_train \
	--do_eval \
	--eval_steps 10 \
	--lora_r ${lora_r} \
	--lora_alpha 4 \
	--lora_modules ${lora_modules} \
	--double_quant \
	--quant_type nf4 \
	--bf16 \
	--bits 4 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type ${lr_scheduler} \
	--dataset ${dataset_path} \
	--dataset_format icbu-tars-multi \
	--model_max_len 3072 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--gradient_accumulation_steps 4 \
	--learning_rate ${lr} \
	--adam_beta2 0.999 \
	--max_grad_norm 0.3 \
	--lora_dropout 0.05 \
	--weight_decay 0.0 \
	--seed 11259 \
	--ddp_find_unused_parameters False \
	--model_id ${model_id} \
	--label_names labels \
	--moe_version $moe_version \
	--expert_num $expert_num \
	--trust_remote_code \
	--gradient_checkpointing






