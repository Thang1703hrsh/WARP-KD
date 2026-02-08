"""
Training script for DistillM-2 knowledge distillation.

This script implements the DistillM-2 training procedure for distilling
a larger teacher model into a smaller student model using adaptive
bidirectional KL divergence.
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime
import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig
)

from transformers import (
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_from_disk, DatasetDict

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import (
    get_optimizer_params,
    get_optimizer_params_peft,
    print_args,
    initialize
)
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm2 import get_distillm2_loss
from rouge_metric import compute_metrics

from peft import PeftModel
from wandb_logger import init_wandb, log_metrics, finish_wandb

torch.set_num_threads(4)


def get_teacher_model(args, device):
    """Load teacher model with optional PEFT adapters."""
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError("Model parallelism not supported for DistillM-2")
    else:
        config.is_model_parallel = False
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                config=config,
                device_map={"": device},
                torch_dtype=torch.float16
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                config=config,
                device_map={"": device},
                torch_dtype=torch.float32
            )
            model = model.half()
        
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError(f"PEFT type {args.peft} not supported")
        else:
            if dist.get_rank() == 0:
                print(f' > Teacher number of parameters: {sum([p.nelement() for p in model.parameters()])}', 
                      flush=True)

    model.eval()
    return model


def compute_distillm2_loss(args, student_model, teacher_model, batch, global_step=None):
    """
    Compute DistillM-2 loss for a batch.
    
    Args:
        args: Training arguments
        student_model: Student model
        teacher_model: Teacher model
        batch: Input batch containing input_ids and attention_mask
        global_step: Current global training step
        
    Returns:
        loss: Scalar loss tensor
    """
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    # Student forward pass
    student_outputs = student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False
    )
    student_logits = student_outputs.logits
    
    # Teacher forward pass (no gradients)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        teacher_logits = teacher_outputs.logits
    
    # Compute DistillM-2 loss
    loss_type = getattr(args, 'distillm2_loss_type', 'distillm_v2')
    gradual_beta = getattr(args, 'gradual_beta', False)
    max_steps = getattr(args, 'total_steps', None)
    
    loss = get_distillm2_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        attention_mask=attention_mask,
        loss_type=loss_type,
        global_step=global_step,
        max_steps=max_steps,
        gradual_beta=gradual_beta,
    )
    
    return loss


def train(args, tokenizer, teacher_model, ds_config):
    """
    Main training loop for DistillM-2.
    
    Args:
        args: Training arguments
        tokenizer: Tokenizer
        teacher_model: Teacher model
        ds_config: DeepSpeed configuration
    """
    device = torch.cuda.current_device()
    
    # Initialize wandb
    if get_rank() == 0 and hasattr(args, 'wandb_project') and args.wandb_project:
        wandb_name = getattr(args, 'wandb_name', None) or f"{args.ckpt_name}-distillm2"
        wandb_config = {
            "type": "distillm2-v2",
            "model": args.ckpt_name,
            "teacher": args.teacher_ckpt_name if hasattr(args, 'teacher_ckpt_name') else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "distillm2_loss_type": getattr(args, 'distillm2_loss_type', 'distillm_v2'),
            "kd_ratio": getattr(args, 'kd_ratio', 0.5),
        }
        wandb_key = getattr(args, 'wandb_key', None)
        base_path = getattr(args, 'base_path', None) or '.'
        init_wandb(args.wandb_project, wandb_name, wandb_config, wandb_key, base_path)
    
    # Get student model
    student_model = get_model(args, device)
    
    # Setup optimizer
    if args.peft is not None:
        optimizer_grouped_parameters = get_optimizer_params_peft(args, student_model)
    else:
        optimizer_grouped_parameters = get_optimizer_params(args, student_model)
    
    # Load DistiLLM-2 formatted datasets (Arrow format with chosen/rejected)
    print_rank("Loading DistiLLM-2 datasets from", args.distillm2_data_dir)
    raw_datasets = load_from_disk(args.distillm2_data_dir)
    
    if get_rank() == 0:
        print(f"Loaded datasets: {list(raw_datasets.keys())}")
        print(f"Train samples: {len(raw_datasets['train'])}")
        if 'test' in raw_datasets:
            print(f"Test samples: {len(raw_datasets['test'])}")
    
    # Use raw datasets directly (like distillm-2-master)
    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['test'] if 'test' in raw_datasets else None
    
    # Extract answers for ROUGE evaluation if available
    eval_answers = None
    if eval_dataset is not None:
        # Try to get answers from the original data
        if hasattr(args, 'prompt_data_dir') and args.prompt_data_dir:
            import json
            dev_path = os.path.join(args.prompt_data_dir, 'dev.jsonl')
            if os.path.exists(dev_path):
                with open(dev_path) as f:
                    raw_data = [json.loads(line) for line in f.readlines()]
                    eval_answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
                    # Take only as many as we have in eval_dataset
                    eval_answers = eval_answers[:len(eval_dataset)]
    
    # Create collate function for tokenization and batching
    def collate_fn(batch):
        """Tokenize and batch data - handles both DistiLLM-2 format and LM format."""
        # Separate items by type since ConcatDataset can mix them in same batch
        distillm2_items = [item for item in batch if 'prompt' in item]
        lm_items = [item for item in batch if 'input_ids' in item]
        
        # Process each type separately and combine
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        # Process DistiLLM-2 data
        if distillm2_items:
            texts = [item['prompt'] + item['chosen'] for item in distillm2_items]
            prompts = [item['prompt'] for item in distillm2_items]
            
            # Tokenize
            encodings = tokenizer(
                texts,
                max_length=args.max_length,
                truncation=True,
                padding='longest',
                return_tensors='pt'
            )
            
            # Get prompt lengths for label masking - tokenize individually
            prompt_lengths = []
            for prompt in prompts:
                prompt_enc = tokenizer(
                    prompt,
                    max_length=args.max_length,
                    truncation=True,
                    add_special_tokens=True,
                )
                prompt_lengths.append(len(prompt_enc['input_ids']))
            
            # Create labels (mask prompt tokens with -100)
            labels = encodings['input_ids'].clone()
            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100
            
            all_input_ids.append(encodings['input_ids'])
            all_attention_mask.append(encodings['attention_mask'])
            all_labels.append(labels)
        
        # Process LM data
        if lm_items:
            input_ids_list = [torch.tensor(item['input_ids'], dtype=torch.long) for item in lm_items]
            
            # Pad to max length in batch
            max_len = min(max(len(ids) for ids in input_ids_list), args.max_length)
            
            lm_input_ids = []
            lm_attention_mask = []
            lm_labels = []
            
            for ids in input_ids_list:
                # Truncate if needed
                ids = ids[:max_len]
                seq_len = len(ids)
                pad_len = max_len - seq_len
                
                # Shift for causal LM: input is [:-1], labels is [1:]
                lm_input_ids.append(torch.cat([ids[:-1], torch.full((pad_len + 1,), tokenizer.pad_token_id, dtype=torch.long)]))
                lm_attention_mask.append(torch.cat([torch.ones(seq_len - 1, dtype=torch.long), torch.zeros(pad_len + 1, dtype=torch.long)]))
                lm_labels.append(torch.cat([ids[1:], torch.full((pad_len + 1,), -100, dtype=torch.long)]))
            
            all_input_ids.append(torch.stack(lm_input_ids))
            all_attention_mask.append(torch.stack(lm_attention_mask))
            all_labels.append(torch.stack(lm_labels))
        
        # Combine all batches
        if len(all_input_ids) == 0:
            raise ValueError("Empty batch received")
        
        # Find max sequence length across all items
        max_seq_len = max(tensor.size(1) for tensor in all_input_ids)
        
        # Pad all tensors to max_seq_len before concatenating
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for input_ids, attention_mask, labels in zip(all_input_ids, all_attention_mask, all_labels):
            current_len = input_ids.size(1)
            if current_len < max_seq_len:
                pad_len = max_seq_len - current_len
                input_ids = torch.cat([input_ids, torch.full((input_ids.size(0), pad_len), tokenizer.pad_token_id, dtype=torch.long)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros((attention_mask.size(0), pad_len), dtype=torch.long)], dim=1)
                labels = torch.cat([labels, torch.full((labels.size(0), pad_len), -100, dtype=torch.long)], dim=1)
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)
        
        # Concatenate along batch dimension
        final_input_ids = torch.cat(padded_input_ids, dim=0)
        final_attention_mask = torch.cat(padded_attention_mask, dim=0)
        final_labels = torch.cat(padded_labels, dim=0)
        
        return {
            'input_ids': final_input_ids,
            'attention_mask': final_attention_mask,
            'labels': final_labels
        }
    
    # Add LM data if provided (for auxiliary language modeling loss with OpenWebText)
    if args.lm_data_dir:
        print_rank("Adding OpenWebText LM data from", args.lm_data_dir)
        # Calculate how many LM samples we need based on kd_ratio
        # kd_ratio = 0.5 means 50% DistiLLM-2 data, 50% LM data
        kd_ratio = args.kd_ratio if hasattr(args, 'kd_ratio') and args.kd_ratio else 0.5
        num_prompt_samples = len(train_dataset)
        num_lm_samples = int(num_prompt_samples * (1 - kd_ratio) / kd_ratio)
        
        print_rank(f"  DistiLLM-2 samples: {num_prompt_samples}")
        print_rank(f"  OpenWebText samples: {num_lm_samples}")
        print_rank(f"  KD ratio: {kd_ratio}")
        
        lm_train_dataset = LMTrainDataset(
            args,
            tokenizer,
            args.lm_data_dir,  # Directory path
            "train",  # Split name
            num=num_lm_samples,
            ratio=kd_ratio,
            rng_sample=random.Random(args.seed)
        )
        
        # Mix prompt and LM datasets using torch's ConcatDataset
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, lm_train_dataset])
        print_rank(f"  Total training samples: {len(train_dataset)}")
    else:
        print_rank("No LM data provided - using DistiLLM-2 data only")
    
    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size()
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    if eval_dataset:
        eval_sampler = DistributedSampler(
            eval_dataset,
            shuffle=False,
            drop_last=False,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size()
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            sampler=eval_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        eval_dataloader = None
    
    # Calculate total steps (needed for lr_scheduler)
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    args.total_steps = total_steps
    
    # Create optimizer for DeepSpeed
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-8,  # Default Adam epsilon
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler (before DeepSpeed init)
    if args.total_iters is None:
        args.total_iters = args.total_steps
    
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")
    
    # Initialize DeepSpeed
    student_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=student_model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        model_parameters=None,
        dist_init_required=False
    )
    
    # Training stats (matching finetune.py)
    step, global_step = 1, 1
    total_loss = 0.0
    total_distil_loss = 0.0
    best_eval_loss = float('inf')
    best_rouge = -1.0
    best_val_iter = -1
    
    if get_rank() == 0:
        print(f"***** Running DistillM-2 Training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num epochs = {args.epochs}")
        print(f"  Batch size per device = {args.batch_size}")
        print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {total_steps}")
        print(f"  Loss type = {getattr(args, 'distillm2_loss_type', 'distillm_v2')}")
    
    # Training loop
    for epoch in range(args.epochs):
        student_model.train()
        train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=(get_rank() != 0)
        )
        
        for it, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Compute loss
            loss = compute_distillm2_loss(
                args, student_model, teacher_model, batch, global_step=global_step
            )
            
            # Backward pass
            student_model.backward(loss)
            student_model.step()
            
            # Update stats (matching finetune.py)
            loss_val = loss.item()
            total_loss += loss_val / (args.log_interval * args.gradient_accumulation_steps)
            total_distil_loss += loss_val / (args.log_interval * args.gradient_accumulation_steps)
            epoch_loss += loss_val
            epoch_steps += 1
            
            # Log metrics (matching finetune.py)
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                progress_bar.set_postfix({"loss": f"{total_loss:.4f}"})
                
                # Log to wandb (matching finetune.py format)
                log_metrics({
                    "train/loss": total_loss,
                    "train/distil_loss": total_distil_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0] if lr_scheduler else args.lr,
                    "train/epoch": epoch,
                }, step=global_step)
                
                total_loss = 0.0
                total_distil_loss = 0.0
            
            # Evaluation (matching finetune.py)
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                if eval_dataloader:
                    eval_loss, eval_results = evaluate(
                        args, tokenizer, student_model, teacher_model, 
                        eval_dataset, eval_dataloader, "dev", global_step, device, eval_answers
                    )
                    
                    if get_rank() == 0:
                        eval_rouge = eval_results.get("rougeL", None)
                        print(f"\nEval Loss: {eval_loss:.4f} | ROUGE-L: {eval_rouge if eval_rouge else 'N/A'}")
                        
                        # Log to wandb (matching finetune.py format)
                        metrics = {"eval/dev_loss": eval_loss}
                        if eval_results:
                            for key, val in eval_results.items():
                                metrics[f"eval/dev_{key}"] = val
                        log_metrics(metrics, step=global_step)
                        
                        # Checkpointing (matching finetune.py)
                        improved = (eval_loss < best_eval_loss)
                        if args.eval_gen:
                            assert eval_rouge is not None and best_rouge is not None
                            improved = (eval_rouge > best_rouge)
                        
                        if args.save and args.save_interval and global_step % args.save_interval == 0 and improved:
                            best_eval_loss = eval_loss
                            if args.eval_gen:
                                best_rouge = eval_rouge
                            best_val_iter = global_step
                            save_dir_path = os.path.join(args.save, str(global_step))
                            if args.model_parallel:
                                raise NotImplementedError
                            else:
                                if getattr(args, 'only_save_last', False):
                                    best_val_iter = -1
                                elif dist.get_rank() == 0:
                                    os.makedirs(save_dir_path, exist_ok=True)
                                    print_rank(f"Model save to {save_dir_path}")
                                    tokenizer.save_pretrained(save_dir_path)
                                    student_model.module.save_pretrained(save_dir_path, safe_serialization=False)
                            dist.barrier()
                
                student_model.train()
            torch.cuda.empty_cache()
            
            # Increment step counters (matching finetune.py)
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
        
        # End of epoch
        if get_rank() == 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} finished. Average loss: {avg_epoch_loss:.4f}")
    
    # Copy best saved checkpoint out to root dir (matching finetune.py)
    if args.save and dist.get_rank() == 0:
        if best_val_iter == -1:
            student_model.module.save_pretrained(args.save, safe_serialization=False)
        else:
            best_ckpt_path = os.path.join(args.save, str(best_val_iter))
            import shutil
            for filename in os.listdir(best_ckpt_path):
                src_path = os.path.join(best_ckpt_path, filename)
                dst_path = os.path.join(args.save, filename)

                if os.path.isfile(src_path):  # only copy files
                    shutil.copy2(src_path, dst_path)  # overwrite if exists
        tokenizer.save_pretrained(args.save)
        print(f"\nTraining completed! Model saved to {args.save}")
        
        # Finish wandb
        finish_wandb()
    
    return student_model, best_val_iter


def evaluate(args, tokenizer, student_model, teacher_model, eval_dataset, eval_dataloader, split, epoch, device, eval_answers=None):
    """
    Evaluate the student model on the validation set.
    
    Args:
        args: Training arguments
        tokenizer: Tokenizer
        student_model: Student model
        teacher_model: Teacher model
        eval_dataset: Raw evaluation dataset
        eval_dataloader: Evaluation dataloader
        split: Split name ("dev" or "test")
        epoch: Current epoch/step
        device: Device to run on
        eval_answers: Ground truth answers for ROUGE evaluation
        
    Returns:
        Tuple of (average evaluation loss, results dict with ROUGE scores)
    """
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    
    student_model.eval()
    total_loss = 0.0
    total_steps = 0
    all_response_ids = []
    
    # Setup generation config (matching finetune.py)
    gen_config_kwargs = {
        'do_sample': getattr(args, 'do_sample', True),
        'top_p': getattr(args, 'top_p', 0.9),
        'top_k': getattr(args, 'top_k', 0),
        'temperature': getattr(args, 'temperature', 1.0),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
        'max_length': args.max_length,
        'min_length': None,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.eos_token_id,
        'return_dict_in_generate': True,
        'output_scores': False
    }
    
    generation_config = GenerationConfig(**gen_config_kwargs)
    
    # Set padding side to left for generation (fixes decoder-only warning)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    with torch.no_grad():
        for it, batch_items in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=(get_rank() != 0))):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch_items.items()}
            
            # Compute loss
            loss = compute_distillm2_loss(
                args, student_model, teacher_model, batch
            )
            
            total_loss += loss.item()
            total_steps += 1
            
            # Generate responses for ROUGE evaluation if answers are available
            if eval_answers:
                # Prepare generation batch from raw dataset items (like finetune.py)
                # Get the raw items for this batch
                start_idx = it * args.eval_batch_size * dp_world_size + dp_rank * args.eval_batch_size
                end_idx = start_idx + batch['input_ids'].size(0)
                
                # Create gen_data by tokenizing just the prompts
                batch_prompts = []
                for idx in range(start_idx, end_idx):
                    if idx < len(eval_dataset):
                        item = eval_dataset[idx]
                        batch_prompts.append(item['prompt'])
                
                if batch_prompts:
                    gen_encodings = tokenizer(
                        batch_prompts,
                        padding='longest',
                        max_length=args.max_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    gen_data = {
                        'input_ids': gen_encodings['input_ids'].to(device),
                        'attention_mask': gen_encodings['attention_mask'].to(device)
                    }
                    
                    max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
                    
                    gen_out = student_model.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens
                    )
                    
                    full_ids = gen_out.sequences
                    
                    # Pad to max_length
                    full_ids = F.pad(
                        full_ids,
                        (0, args.max_length - full_ids.shape[1]),
                        value=tokenizer.pad_token_id,
                    )
                    
                    # Extract generated part only
                    response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                    all_response_ids.append(response_ids)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Gather losses from all processes
    total_loss = torch.tensor(total_loss, device=device)
    total_steps = torch.tensor(total_steps, device=device)
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM, group=dp_group)
    
    avg_loss = total_loss.item() / total_steps.item()
    
    # Compute ROUGE metrics if we generated responses
    results = {}
    if all_response_ids and eval_answers:
        # Pad all response_ids to same size before concatenating
        max_response_len = max(r.size(1) for r in all_response_ids)
        padded_response_ids = []
        for r in all_response_ids:
            if r.size(1) < max_response_len:
                r = F.pad(r, (0, max_response_len - r.size(1)), value=tokenizer.pad_token_id)
            padded_response_ids.append(r)
        
        all_response_ids = torch.cat(padded_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
        responses = responses[:len(eval_answers)]
        
        # Compute ROUGE metrics
        results = compute_metrics(responses, eval_answers)
        
        # Save generated responses
        if get_rank() == 0:
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
    
    if get_rank() == 0:
        log_str = f"{split} | avg_loss: {avg_loss:.4f}"
        if results:
            log_str += f" | {results}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    
    return avg_loss, results


def main():
    """Main entry point for DistillM-2 training."""
    args = get_args()
    initialize(args)
    
    device = torch.cuda.current_device()
    
    # Create save directory
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
    args.deepspeed_config = None
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    # Load teacher and tokenizer
    teacher_model = get_teacher_model(args, device)
    tokenizer = get_tokenizer(args)
    
    # Train
    if args.do_train:
        model, best_val_iter = train(
            args=args,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            ds_config=ds_config
        )


if __name__ == "__main__":
    main()
