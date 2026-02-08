#!/usr/bin/env python
# coding=utf-8
"""
DistiLLM-2 Evaluation Script

Evaluates DistiLLM-2 distilled models on the Dolly dataset with ROUGE-L metrics.
Compatible with Contra-KD's evaluation framework.
"""

import time
import os
import torch
import torch.distributed as dist
import deepspeed
import json
import numpy as np

from arguments import get_args
from utils import initialize, print_args, print_rank, save_rank
from utils import get_tokenizer, get_model
from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rouge_metric import compute_metrics
from utils import all_gather

import nltk
nltk.download("punkt", quiet=True)

torch.set_num_threads(4)


def setup_model(args, ds_config, device):
    """Setup model with DeepSpeed for evaluation."""
    model = get_model(args, device)
    optimizer, lr_scheduler = None, None
        
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model


def prepare_dataset(args, tokenizer):
    """Prepare Dolly dataset for evaluation."""
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    return data


def run_evaluation(args, tokenizer, model, dataset: PromptDataset, epoch, device):
    """
    Run evaluation on the dataset.
    
    Returns:
        mean_lm_loss: Average language modeling loss
        query_ids: Input prompts
        response_ids: Generated responses
    """
    collate_fn = dataset.collate
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []
    
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating DistiLLM-2 on {args.data_names}", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example Input ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            # Compute LM loss
            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
            lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss)

            # Generate responses
            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):]  # remove prompt
            
            query_ids = F.pad(query_ids, (args.max_prompt_length-query_ids.size(1), 0, 0, 0), value=tokenizer.pad_token_id)
            response_ids = F.pad(response_ids, (0, args.max_length-args.max_prompt_length-response_ids.size(1), 0, 0), value=tokenizer.pad_token_id)
            
            all_query_ids.append(query_ids)
            all_response_ids.append(response_ids)

    # Aggregate losses
    all_lm_losses = torch.cat(all_lm_losses)
    mean_lm_loss = all_lm_losses.mean()
    dist.all_reduce(mean_lm_loss, dist.ReduceOp.SUM, group=dp_group)
    mean_lm_loss = mean_lm_loss.item() / dp_world_size
        
    # Gather all outputs
    all_query_ids = torch.cat(all_query_ids)
    all_query_ids = all_gather(all_query_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_query_ids = all_query_ids.view(-1, all_query_ids.size(-1))
    all_query_ids = all_query_ids[:len(dataset)]
    
    all_response_ids = torch.cat(all_response_ids)
    all_response_ids = all_gather(all_response_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    all_response_ids = all_response_ids[:len(dataset)]
        
    return mean_lm_loss, all_query_ids, all_response_ids


def evaluate_and_save(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    """
    Run evaluation and save results with ROUGE-L metrics.
    """
    lm_loss, query_ids, response_ids = run_evaluation(args, tokenizer, model, dataset, epoch, device)
    
    # Decode to text
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    # Save predictions
    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    # Process responses
    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:    
        for q, r in zip(query_strs, response_strs):
            # Remove prompt from response
            response_text = r[len(q):]
            # Remove end token if present
            idx = response_text.find("<|endoftext|>")
            if idx >= 0:
                response_text = response_text[:idx]
            
            cleaned_response = response_text.replace("<n>", "\n").strip()
            f.write(json.dumps({"text": cleaned_response}) + "\n")
            all_responses.append(cleaned_response)
    
    # Compute ROUGE-L and Exact Match metrics
    gen_metrics = compute_metrics(all_responses, dataset.answers)
    
    # Compute average generation length
    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    # Log results
    log_str = (
        f"{split} | Dataset: {args.data_names} | "
        f"ROUGE-L: {gen_metrics['rougeL']:.4f} | "
        f"Exact Match: {gen_metrics['exact_match']:.4f} | "
        f"LM Loss: {lm_loss:.4f} | "
        f"Avg Gen Length: {mean_gen_length:.2f}"
    )
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    
    # Save metrics to JSON
    if dist.get_rank() == 0:
        metrics_dict = {
            "split": split,
            "dataset": args.data_names,
            "rougeL": gen_metrics['rougeL'],
            "exact_match": gen_metrics['exact_match'],
            "lm_loss": lm_loss,
            "avg_generation_length": mean_gen_length,
            "num_samples": len(all_responses)
        }
        with open(os.path.join(args.save, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        print_rank(f"\nMetrics saved to {os.path.join(args.save, 'metrics.json')}")


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    
    # Set evaluation mode
    args.do_train = False
    args.do_eval = True
    
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" DistiLLM-2 Evaluation at {cur_time} " + "="*30, 
              os.path.join(args.save, "log.txt"))
    
    # Load DeepSpeed config
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    # Disable ZeRO for evaluation
    ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"] 
    args.deepspeed_config = None

    # Get tokenizer and dataset
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)
    
    # Setup model
    model = setup_model(args, ds_config, device)
    
    # Run evaluation
    evaluate_and_save(args, tokenizer, model, dataset["test"], "test", 0, device)
    
    print_rank("\n" + "="*50)
    print_rank("DistiLLM-2 Evaluation Complete!")
    print_rank("="*50)


if __name__ == "__main__":
    main()
