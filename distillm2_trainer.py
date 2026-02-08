"""
DistiLLM-2 Trainer with Contra-KD Integration

This module implements the DistiLLM-2 training framework integrated with
the Contra-KD architecture, enabling contrastive distillation with 
adaptive mixing coefficients.
"""

import os
import json
import math
from time import time
from typing import Optional, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from utils import print_rank, save_rank, get_rank, all_gather, save_parallel
from distillm2.losses import get_distillm2_loss


class DistiLLM2ContraTrainer:
    """
    Trainer for DistiLLM-2 with Contra-KD integration.
    
    This trainer implements the contrastive distillation approach from DistiLLM-2
    within the Contra-KD framework, supporting:
    - DistiLLM-v1 and DistiLLM-v2 loss functions
    - Adaptive mixing coefficients
    - Gradual beta scheduling
    - DeepSpeed integration
    """
    
    def __init__(
        self,
        args,
        tokenizer: AutoTokenizer,
        teacher_model: nn.Module,
        ds_config: dict,
        train_dataset,
        eval_dataset,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.ds_config = ds_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.device = torch.cuda.current_device()
        self.max_length = args.max_length
        self.max_prompt_length = getattr(args, 'max_prompt_length', 128)
        
        # DistiLLM-2 specific configs
        self.loss_type = getattr(args, 'distillm2_loss_type', 'distillm_v2')
        self.beta = getattr(args, 'distillm2_beta', 0.1)
        self.base_alpha_1 = getattr(args, 'distillm2_base_alpha_1', 0.1)
        self.base_alpha_2 = getattr(args, 'distillm2_base_alpha_2', 0.1)
        self.gradual_beta = getattr(args, 'distillm2_gradual_beta', False)
        
        # Training state
        self.global_step = 0
        self.total_steps = args.total_iters if args.total_iters else args.train_iters_per_epoch * args.training_epochs
        
        # Distributed training setup
        if args.model_parallel:
            raise NotImplementedError("Model parallel not yet supported")
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None
        
        # Initialize student model
        self.model = self._load_student_model()
        
        # Setup optimization
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.model, self.optimizer, self.scheduler = self._setup_deepspeed()
        
        # Metrics tracking
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.logp_logq = None  # For adaptive alpha in v2
        self.logq_logp = None  # For adaptive alpha in v2
        
    def _load_student_model(self):
        """Load the student model for distillation."""
        config_path = self.args.model_path if hasattr(self.args, 'model_path') else self.args.load
        
        model = AutoModelForCausalLM.from_pretrained(
            config_path,
            torch_dtype=torch.float16 if not self.args.fp32 else torch.float32,
            use_cache=False,
        )
        
        if dist.get_rank() == 0:
            print(f' > Student model parameters: {sum([p.nelement() for p in model.parameters()]) / 1e6:.2f}M')
        
        return model
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            betas=(0.9, 0.95),
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.args.lr_decay_style == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_iters,
                num_training_steps=self.total_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_iters,
            )
        
        return scheduler
    
    def _setup_deepspeed(self):
        """Initialize DeepSpeed engine."""
        import deepspeed
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=self.ds_config,
        )
        
        return model_engine, optimizer, scheduler
    
    def tokenize_row(self, example):
        """
        Tokenize a single example for DistiLLM-2 training.
        
        Supports two formats:
        1. DistiLLM-2 format: {prompt, chosen, rejected}
        2. Contra-KD Dolly format: {prompt, output}
        """
        # Handle both DistiLLM-2 format and Contra-KD format
        if 'chosen' in example and 'rejected' in example:
            # DistiLLM-2 format
            prompt = example['prompt']
            chosen = example['chosen']
            rejected = example['rejected']
        else:
            # Contra-KD Dolly format - use prompt as is
            # We'll generate teacher/student responses during data generation
            prompt = example['prompt'].replace('<n>', '\n')
            chosen = example.get('output', '')  # Teacher output
            rejected = example.get('student_output', chosen)  # Student output (fallback to teacher)
        
        prompt = prompt
        chosen = chosen
        rejected = rejected
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
        )
        
        # Tokenize chosen (teacher) response
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length - self.max_prompt_length,
            add_special_tokens=False,
        )
        
        # Tokenize rejected (student) response
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length - self.max_prompt_length,
            add_special_tokens=False,
        )
        
        # Combine prompt + response
        chosen_input_ids = prompt_tokens['input_ids'] + chosen_tokens['input_ids']
        rejected_input_ids = prompt_tokens['input_ids'] + rejected_tokens['input_ids']
        
        # Create labels (mask prompt tokens)
        prompt_len = len(prompt_tokens['input_ids'])
        chosen_labels = [-100] * prompt_len + chosen_tokens['input_ids']
        rejected_labels = [-100] * prompt_len + rejected_tokens['input_ids']
        
        return {
            'chosen_input_ids': chosen_input_ids,
            'rejected_input_ids': rejected_input_ids,
            'chosen_labels': chosen_labels,
            'rejected_labels': rejected_labels,
            'prompt': prompt,
        }
    
    def collate_fn(self, batch):
        """Collate batch for training."""
        # Find max lengths
        max_chosen_len = max(len(ex['chosen_input_ids']) for ex in batch)
        max_rejected_len = max(len(ex['rejected_input_ids']) for ex in batch)
        max_len = max(max_chosen_len, max_rejected_len)
        
        # Pad sequences
        chosen_input_ids = []
        chosen_labels = []
        chosen_attention_mask = []
        rejected_input_ids = []
        rejected_labels = []
        rejected_attention_mask = []
        
        for ex in batch:
            # Chosen
            chosen_pad_len = max_len - len(ex['chosen_input_ids'])
            chosen_input_ids.append(ex['chosen_input_ids'] + [self.tokenizer.pad_token_id] * chosen_pad_len)
            chosen_labels.append(ex['chosen_labels'] + [-100] * chosen_pad_len)
            chosen_attention_mask.append([1] * len(ex['chosen_input_ids']) + [0] * chosen_pad_len)
            
            # Rejected
            rejected_pad_len = max_len - len(ex['rejected_input_ids'])
            rejected_input_ids.append(ex['rejected_input_ids'] + [self.tokenizer.pad_token_id] * rejected_pad_len)
            rejected_labels.append(ex['rejected_labels'] + [-100] * rejected_pad_len)
            rejected_attention_mask.append([1] * len(ex['rejected_input_ids']) + [0] * rejected_pad_len)
        
        # Concatenate chosen and rejected
        concatenated_input_ids = chosen_input_ids + rejected_input_ids
        concatenated_labels = chosen_labels + rejected_labels
        concatenated_attention_mask = chosen_attention_mask + rejected_attention_mask
        
        return {
            'input_ids': torch.tensor(concatenated_input_ids, dtype=torch.long),
            'labels': torch.tensor(concatenated_labels, dtype=torch.long),
            'attention_mask': torch.tensor(concatenated_attention_mask, dtype=torch.long),
            'batch_size': len(batch),
        }
    
    def compute_loss(self, batch):
        """
        Compute DistiLLM-2 loss for a batch.
        
        The batch contains concatenated chosen (teacher) and rejected (student) samples.
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        batch_size = batch['batch_size']
        
        # Split into chosen and rejected
        chosen_input_ids = input_ids[:batch_size]
        rejected_input_ids = input_ids[batch_size:]
        chosen_labels = labels[:batch_size]
        rejected_labels = labels[batch_size:]
        chosen_attention_mask = attention_mask[:batch_size]
        rejected_attention_mask = attention_mask[batch_size:]
        
        # Get student logits for both chosen and rejected
        student_outputs_chosen = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            use_cache=False,
        )
        student_logits_chosen = student_outputs_chosen.logits
        
        student_outputs_rejected = self.model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            use_cache=False,
        )
        student_logits_rejected = student_outputs_rejected.logits
        
        # Get teacher logits for both chosen and rejected
        with torch.no_grad():
            teacher_outputs_chosen = self.teacher_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                use_cache=False,
            )
            teacher_logits_chosen = teacher_outputs_chosen.logits
            
            teacher_outputs_rejected = self.teacher_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                use_cache=False,
            )
            teacher_logits_rejected = teacher_outputs_rejected.logits
        
        # Compute DistiLLM-2 loss for chosen samples
        loss_chosen = get_distillm2_loss(
            student_logits=student_logits_chosen,
            teacher_logits=teacher_logits_chosen,
            labels=chosen_labels,
            attention_mask=chosen_attention_mask,
            loss_type=self.loss_type,
            global_step=self.global_step,
            max_steps=self.total_steps,
            gradual_beta=self.gradual_beta,
        )
        
        # Compute DistiLLM-2 loss for rejected samples
        loss_rejected = get_distillm2_loss(
            student_logits=student_logits_rejected,
            teacher_logits=teacher_logits_rejected,
            labels=rejected_labels,
            attention_mask=rejected_attention_mask,
            loss_type=self.loss_type,
            global_step=self.global_step,
            max_steps=self.total_steps,
            gradual_beta=self.gradual_beta,
        )
        
        # Total loss (average of chosen and rejected)
        loss = (loss_chosen + loss_rejected) / 2
        
        # Track metrics
        metrics = {
            'loss': loss.item(),
            'loss_chosen': loss_chosen.item(),
            'loss_rejected': loss_rejected.item(),
        }
        
        return loss, metrics
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        loss, metrics = self.compute_loss(batch)
        
        # Backward pass
        self.model.backward(loss)
        self.model.step()
        
        self.global_step += 1
        
        return metrics
    
    def train(self):
        """Main training loop."""
        # Prepare data
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
        )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            collate_fn=self.collate_fn,
            num_workers=0,
        )
        
        # Training loop
        epoch = 0
        total_loss = 0
        log_metrics = defaultdict(float)
        
        if dist.get_rank() == 0:
            print(f"Starting training for {self.args.training_epochs} epochs")
            print(f"Total steps: {self.total_steps}")
            print(f"Steps per epoch: {len(train_dataloader)}")
        
        while epoch < self.args.training_epochs:
            epoch += 1
            train_sampler.set_epoch(epoch)
            
            if dist.get_rank() == 0:
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            else:
                pbar = train_dataloader
            
            for step, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                
                total_loss += metrics['loss']
                for k, v in metrics.items():
                    log_metrics[k] += v
                
                # Logging
                if (step + 1) % self.args.log_interval == 0:
                    avg_loss = total_loss / self.args.log_interval
                    
                    if dist.get_rank() == 0:
                        log_str = f"Epoch {epoch} Step {self.global_step}: loss={avg_loss:.4f}"
                        for k, v in log_metrics.items():
                            if k != 'loss':
                                log_str += f", {k}={v/self.args.log_interval:.4f}"
                        print_rank(log_str)
                    
                    total_loss = 0
                    log_metrics = defaultdict(float)
                
                # Checkpointing
                if self.args.save_interval > 0 and (step + 1) % self.args.save_interval == 0:
                    self._save_checkpoint(epoch, step)
                
                # Check if we've reached total_iters
                if self.args.total_iters and self.global_step >= self.args.total_iters:
                    break
            
            # End of epoch checkpoint
            if self.args.save_interval > 0:
                self._save_checkpoint(epoch, -1)
            
            # Early stopping if total_iters reached
            if self.args.total_iters and self.global_step >= self.args.total_iters:
                if dist.get_rank() == 0:
                    print_rank(f"Reached total_iters={self.args.total_iters}, stopping training")
                break
        
        if dist.get_rank() == 0:
            print_rank("Training completed!")
    
    def _save_checkpoint(self, epoch, step):
        """Save model checkpoint."""
        if dist.get_rank() == 0:
            if step == -1:
                save_path = os.path.join(self.args.save, f"epoch_{epoch}")
            else:
                save_path = os.path.join(self.args.save, f"epoch_{epoch}_step_{step}")
            
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            self.model.save_checkpoint(save_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            # Save training state
            state = {
                'epoch': epoch,
                'global_step': self.global_step,
                'args': vars(self.args),
            }
            with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
                json.dump(state, f, indent=2)
            
            print_rank(f"Checkpoint saved to {save_path}")
