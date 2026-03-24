import os
import torch
from os.path import exists, join, isdir
from torch.distributed import get_rank



def rank0_print(*args, **kwargs):
    try:
        if get_rank() == 0:
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)



def print_trainable_parameters(args, model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    rank0_print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
    params_on_cpu = [n for n, param in model.named_parameters() if param.device == torch.device("cpu")]
    rank0_print("No params on cpu" if not params_on_cpu else f"{params_on_cpu} on cpu.")



def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True 
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        rank0_print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed
    return None, False 

