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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model, get_distillation_schedule

# from distillm import forward_kl, reverse_kl, js_distance, tv_distance
# from distillm import skewed_forward_kl, skewed_reverse_kl
# from distillm import SampleGenerator, ReplayBuffer

from distillm.velocity_field import VelocityField
from distillm.projector import Projector
from distillm.losses import velocity_field_loss

from rouge_metric import compute_metrics

from peft import PeftModel

from wandb_logger import init_wandb, log_metrics, finish_wandb

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
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

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    
    if "distillm2" in args.type:
        # Load Arrow datasets for DistiLLM-2
        from datasets import load_from_disk
        print_rank(f"Loading DistiLLM-2 Arrow dataset from {args.data_dir}")
        raw_datasets = load_from_disk(args.data_dir)
        
        # Create wrapper class with collate method
        class DistiLLM2Dataset:
            def __init__(self, dataset, tokenizer, max_length, max_prompt_length=256):
                self.dataset = dataset
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.max_prompt_length = max_prompt_length
                # Extract answers for evaluation
                self.answers = [[item['chosen']] for item in dataset]
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                sample = self.dataset[idx]
                sample["idx"] = idx
                return sample
            
            def move_to_device(self, model_data, no_model_data, gen_data, device):
                for k in model_data:
                    model_data[k] = model_data[k].to(device)

                for k in no_model_data:
                    if isinstance(no_model_data[k], torch.Tensor):
                        no_model_data[k] = no_model_data[k].to(device)

                for k in gen_data:
                    gen_data[k] = gen_data[k].to(device)

                return model_data, no_model_data, gen_data
            
            def collate(self, examples):
                # Tokenize prompts and responses (chosen = teacher, rejected = student)
                prompts = [ex['prompt'] for ex in examples]
                chosen = [ex['chosen'] for ex in examples]
                rejected = [ex['rejected'] for ex in examples]
                examples_idx = [ex['idx'] for ex in examples]
                
                # Tokenize inputs - use max_prompt_length for prompts to ensure consistent gen_data size
                prompt_tokens = self.tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_prompt_length)
                
                # Tokenize full sequences (prompt + response)
                chosen_full = [p + c for p, c in zip(prompts, chosen)]
                rejected_full = [p + r for p, r in zip(prompts, rejected)]
                
                chosen_tokens = self.tokenizer(chosen_full, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                rejected_tokens = self.tokenizer(rejected_full, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                
                # Find the max length between chosen and rejected to pad them equally
                max_len = max(chosen_tokens['input_ids'].shape[1], rejected_tokens['input_ids'].shape[1])
                
                # Pad chosen and rejected to the same length
                def pad_to_length(tensor, target_length, pad_value):
                    if tensor.shape[1] >= target_length:
                        return tensor[:, :target_length]
                    padding = torch.full((tensor.shape[0], target_length - tensor.shape[1]), pad_value, dtype=tensor.dtype)
                    return torch.cat([tensor, padding], dim=1)
                
                chosen_input_ids = pad_to_length(chosen_tokens['input_ids'], max_len, self.tokenizer.pad_token_id)
                chosen_attention_mask = pad_to_length(chosen_tokens['attention_mask'], max_len, 0)
                rejected_input_ids = pad_to_length(rejected_tokens['input_ids'], max_len, self.tokenizer.pad_token_id)
                rejected_attention_mask = pad_to_length(rejected_tokens['attention_mask'], max_len, 0)
                
                # Create labels by masking prompt tokens with -100
                chosen_labels = chosen_input_ids.clone()
                rejected_labels = rejected_input_ids.clone()
                
                # Mask prompt tokens in labels (only compute loss on response)
                for i in range(len(examples)):
                    prompt_len = (prompt_tokens['attention_mask'][i] == 1).sum().item()
                    # Find actual prompt length in the full sequence (accounting for tokenizer merging)
                    # Use a safe approach: mask first max_prompt_length tokens
                    chosen_labels[i, :prompt_len] = -100
                    rejected_labels[i, :prompt_len] = -100
                
                # Concatenate chosen and rejected for efficient forward pass
                concatenated_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
                concatenated_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
                concatenated_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
                
                # Create model_batch and no_model_batch with concatenated inputs
                model_batch = {
                    'input_ids': concatenated_input_ids,
                    'attention_mask': concatenated_attention_mask,
                }
                
                no_model_batch = {
                    'label': concatenated_labels,
                    'attention_mask': concatenated_attention_mask,
                    'batch_size': len(examples),  # Store original batch size to split chosen/rejected later
                    'examples_idx': examples_idx
                }
                
                # Create gen_data for evaluation - use fixed max_prompt_length to match LMTrainDataset format
                gen_data = {
                    'input_ids': prompt_tokens['input_ids'],
                    'attention_mask': prompt_tokens['attention_mask'],
                }
                
                return model_batch, no_model_batch, gen_data
        
        if args.do_train:
            data["train"] = DistiLLM2Dataset(raw_datasets["train"], tokenizer, args.max_length, args.max_prompt_length)
            # data["dev"] = DistiLLM2Dataset(raw_datasets["test"] if "test" in raw_datasets else raw_datasets["train"], tokenizer, args.max_length, args.max_prompt_length)
            print_rank("train num", len(data["train"]))
            # data["dev"] = LMTrainDataset(args, tokenizer, args.gt_data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        elif args.do_eval:
            data["test"] = DistiLLM2Dataset(raw_datasets["test"] if "test" in raw_datasets else raw_datasets["train"], tokenizer, args.max_length, args.max_prompt_length)
    else:
        # Use standard LMTrainDataset for other distillation methods
        if args.do_train:
            data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            print_rank("train num", len(data["train"]))
            data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        elif args.do_eval:
            data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        else:
            raise ValueError("Do train and do eval must set one")
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def train(
    args, 
    tokenizer: AutoTokenizer, 
    model: deepspeed.DeepSpeedEngine, 
    optimizer: AdamW, 
    lr_scheduler, 
    dataset, 
    device, 
    teacher_model=None,
    velocity_field: VelocityField | None = None,
    projector: Projector | None = None
):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        # dp_group = None
        # loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    teacher_schedule, student_schedule = get_distillation_schedule(
        args.num_teacher_layers,
        args.num_student_layers,
        args.num_distill_layers
    )

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        # model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            torch.cuda.synchronize()
            st_time = time.time()

            with torch.no_grad():
                if teacher_model is not None:
                    dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, f"cuda:{args.teacher_device}")
                    teacher_outputs = teacher_model(**model_batch, use_cache=False, output_hidden_states=True)
                else:
                    raise NotImplementedError
                dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, f"cuda:{args.student_device}")
                outputs = model(**model_batch, use_cache=False, output_hidden_states=True)
            # velocity field loss computation here
            loss = velocity_field_loss(
                outputs.hidden_states,
                teacher_outputs.hidden_states,
                velocity_field,
                projector,
                teacher_schedule,
                student_schedule,
                model_batch["attention_mask"],
                args.student_device
            )

            loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            global_loss = loss.item()

            global_distil_loss = 0
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss / (args.log_interval * args.gradient_accumulation_steps)
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss,
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                # print_rank("*" * 100)
                print_rank(log_str)
                # print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                
                # Log to wandb (only rank 0)
                if dist.get_rank() == 0:
                    log_metrics({
                        "velocity_field/loss": total_loss,
                        "velocity_field/lr": lr_scheduler.get_last_lr()[0],
                        "velocity_field/epoch": epoch,
                    }, step=global_step)
                
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
            lr_scheduler.step()
            
    optimizer.zero_grad()
    return velocity_field, projector


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    # Initialize wandb (only on rank 0) - reads from wandb_config.yaml if key not provided
    if dist.get_rank() == 0:
        wandb_name = args.wandb_name or f"velocity-field-{args.ckpt_name}"
        wandb_config = {
            "type": "velocity_field",
            "student": args.ckpt_name,
            "teacher": args.teacher_model_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "velocity_n_layers": args.velocity_n_layers,
            "velocity_d_model": args.velocity_d_model,
        }
        # Pass wandb_key (can be None, will auto-load from YAML)
        init_wandb(args.wandb_project, wandb_name, wandb_config, args.wandb_key, args.base_path)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    # setup student model
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, args.student_device, set_optim=False)
    model.eval()
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    # setup teacher model
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, args.teacher_device)
    else:
        teacher_model = None
    
    # setup velocity field and projector
    velocity_field = VelocityField(
        d_input=args.d_teacher,
        d_model=args.velocity_d_model,
        num_distill_layers=args.num_distill_layers,
        n_layers=args.velocity_n_layers
    ).to(f"cuda:{args.student_device}")
    projector = Projector(
        d_student=args.d_student,
        d_teacher=args.d_teacher
    ).to(f"cuda:{args.student_device}")
    velocity_field.train()
    projector.train()
    
    optimizer = get_optimizer(args, velocity_field)
    assert type(optimizer) is torch.optim.AdamW
    optimizer.add_param_group({'params': projector.parameters()})
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    
    if args.do_train:
        velocity_field, projector = train(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model, velocity_field, projector)
    
    if args.save:
        save_dir_path = os.path.join(args.save)
        if args.model_parallel:
            raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                os.makedirs(save_dir_path, exist_ok=True)
                print_rank(f"Model save to {save_dir_path}")
                torch.save(velocity_field.state_dict(), os.path.join(save_dir_path, "velocity_field.pth"))
                torch.save(projector.state_dict(), os.path.join(save_dir_path, "projector.pth"))
        dist.barrier()
    
    # Finish wandb
    if dist.get_rank() == 0:
        finish_wandb()
        
    
if __name__ == "__main__":
    main()