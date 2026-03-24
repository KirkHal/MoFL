#!/bin/bash


foundation=llama-2-7b
model_name_or_path= #CUSTOM_MODEL_PATH
dataset=puffin
conv_temp=llm2
moe_version=5
expert_num=2
lora_r=4
custom= #CUSTOM_TAG
model_id=${foundation}_${dataset}_${custom}_${conv_temp}_MoEV${moe_version}_exp${expert_num}
out_dir=trained_models/${model_id}


script_dir=$(cd "$(dirname "$0")" && pwd)
moe_state_path=${script_dir}/${out_dir}/quant_state_dict.pth


cd big_refactor                                                             
export HF_DATASETS_CACHE=Datasets_cache/huggingface/datasets
export HF_HOME=Datasets_cache/huggingface/    




batch_size=2

tasks=("arc_easy" "arc_challenge")
# tasks:("arc_easy" "arc_challenge" "hellaswag" "mmlu" "winogrande" "gsm8k" "triviaqa" "openbookqa" "mathqa" "toxigen")

for task in "${tasks[@]}"; do
    echo "TASK = ${task}"

    case $task in
        arc_challenge|arc_easy) num_fewshot=25 ;;
        hellaswag)     num_fewshot=10 ;;
        openbookqa)    num_fewshot=0 ;;
        *)             num_fewshot=5 ;;        
    esac
    echo "num_fewshot = ${num_fewshot}"


    export NCCL_ASYNC_ERROR_HANDLING=1
    port=$((20000 + RANDOM % 100))

    CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes=2 --main_process_port=${port} -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name_or_path,peft=moe,expert_nums=$expert_num,moe_state_path=$moe_state_path,moe_version=$moe_version,lora_r=$lora_r,trust_remote_code=True \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --batch_size ${batch_size} \
        --output_path "./eval_results/${model_id}/${task}_${num_fewshot}-shot.json"
done