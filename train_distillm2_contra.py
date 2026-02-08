#!/usr/bin/env python
# coding=utf-8
"""
DistiLLM-2 + Contra-KD Integration Training Script

This script integrates DistiLLM-2's contrastive distillation approach 
with the Contra-KD framework for knowledge distillation of LLMs.
"""

import torch
import os
import json
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
from datasets import load_dataset, DatasetDict
from peft import PeftModel

from arguments import get_args
from utils import print_args, initialize, get_tokenizer
from distillm2_trainer import DistiLLM2ContraTrainer


def get_teacher_model(args, device):
    """Load teacher model for distillation."""
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    
    if args.model_parallel:
        raise NotImplementedError("Model parallel not yet supported for DistiLLM-2")
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=torch.float16
        )

        if args.peft is not None:
            if args.peft == "lora":
                assert args.teacher_peft_path is not None
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError(f"PEFT type {args.peft} not supported")
        else:
            if dist.get_rank() == 0:
                print(' > Teacher model parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    return model


def load_distillm2_data(args):
    """
    Load paired teacher-student data for DistiLLM-2 training.
    
    Supports:
    1. DistiLLM-2 format: {prompt, chosen, rejected}
    2. Contra-KD Dolly format: {prompt, output} - will use prompt_data_dir
    """
    if args.distillm2_data_dir is not None:
        # Load DistiLLM-2 format data
        if os.path.isdir(args.distillm2_data_dir):
            # Check if it's Arrow format
            if os.path.exists(os.path.join(args.distillm2_data_dir, 'dataset_dict.json')):
                raw_datasets = DatasetDict.load_from_disk(args.distillm2_data_dir)
            else:
                # Load from JSON files
                raw_datasets = DatasetDict({
                    'train': load_dataset('json', data_files=os.path.join(args.distillm2_data_dir, 'train.json'), split='train'),
                    'test': load_dataset('json', data_files=os.path.join(args.distillm2_data_dir, 'dev.json'), split='train'),
                })
        else:
            raw_datasets = DatasetDict({
                'train': load_dataset('json', data_files=os.path.join(args.distillm2_data_dir, 'train.json'), split='train'),
                'test': load_dataset('json', data_files=os.path.join(args.distillm2_data_dir, 'dev.json'), split='train'),
            })
    elif hasattr(args, 'prompt_data_dir') and args.prompt_data_dir is not None:
        # Use Contra-KD Dolly format from prompt_data_dir
        train_file = os.path.join(args.prompt_data_dir, 'train.jsonl')
        valid_file = os.path.join(args.prompt_data_dir, 'valid.jsonl')
        
        raw_datasets = DatasetDict({
            'train': load_dataset('json', data_files=train_file, split='train'),
            'test': load_dataset('json', data_files=valid_file, split='train'),
        })
    else:
        raise ValueError("Please specify either --distillm2-data-dir or --prompt-data-dir for DistiLLM-2 training")
    
    return raw_datasets


def main():
    args = get_args()
    
    # Add DistiLLM-2 specific args if not present
    if not hasattr(args, 'distillm2_loss_type'):
        args.distillm2_loss_type = 'distillm_v2'  # Default to v2
    if not hasattr(args, 'distillm2_beta'):
        args.distillm2_beta = 0.1
    if not hasattr(args, 'distillm2_base_alpha_1'):
        args.distillm2_base_alpha_1 = 0.1
    if not hasattr(args, 'distillm2_base_alpha_2'):
        args.distillm2_base_alpha_2 = 0.1
    if not hasattr(args, 'distillm2_gradual_beta'):
        args.distillm2_gradual_beta = False
    if not hasattr(args, 'max_prompt_length'):
        args.max_prompt_length = 128
    
    initialize(args)
    device = torch.cuda.current_device()
    
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Load DeepSpeed config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    # Load models
    teacher_model = get_teacher_model(args, device)
    tokenizer = get_tokenizer(args)
    
    # Load DistiLLM-2 paired data
    distillm2_datasets = load_distillm2_data(args)
    
    if dist.get_rank() == 0:
        print(f"Loaded DistiLLM-2 datasets:")
        print(f"  Train: {len(distillm2_datasets['train'])} samples")
        print(f"  Test: {len(distillm2_datasets['test'])} samples")
    
    # Initialize trainer
    trainer = DistiLLM2ContraTrainer(
        args=args,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        ds_config=ds_config,
        train_dataset=distillm2_datasets['train'],
        eval_dataset=distillm2_datasets['test'],
    )
    
    # Train
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("Starting DistiLLM-2 + Contra-KD Training")
        print("="*80 + "\n")
    
    trainer.train()
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
