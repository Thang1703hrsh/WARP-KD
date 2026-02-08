"""
Generate responses from teacher and student models for DistiLLM-2 training.

This script generates responses using vLLM for efficient inference,
creating the paired data needed for contrastive distillation.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

try:
    from vllm import SamplingParams, LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")


def generate_with_vllm(
    model_path: str,
    prompts: list,
    output_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    seed: int = 42,
    tensor_parallel_size: int = 1,
):
    """Generate responses using vLLM."""
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for generation. Install with: pip install vllm")
    
    # Check if path contains LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora:
        # Load adapter config to get base model path
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get('base_model_name_or_path')
        
        print(f"Detected LoRA adapter at {model_path}")
        print(f"Base model: {base_model_path}")
        
        # Initialize LLM with base model and enable LoRA
        llm = LLM(
            model=base_model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=True,
        )
        tokenizer = llm.get_tokenizer()
    else:
        # Initialize LLM normally
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
        )
        tokenizer = llm.get_tokenizer()
    
    # Format conversations (handle models with and without chat templates)
    conversations = []
    for prompt in prompts:
        # Check if model has a chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Use chat template for instruction-tuned models
            conversation = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use raw prompt for base models (e.g., GPT-2)
            conversation = prompt
        conversations.append(conversation)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed
    )
    
    # Generate with LoRA if applicable
    print(f"Generating {len(prompts)} responses...")
    if is_lora:
        from vllm.lora.request import LoRARequest
        outputs = llm.generate(
            conversations, 
            sampling_params,
            lora_request=LoRARequest("adapter", 1, model_path)
        )
    else:
        outputs = llm.generate(conversations, sampling_params)
    
    # Save outputs
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_data.append({
            'prompt': prompts[i],
            'format_prompt': prompt,
            'generated_text': generated_text,
        })
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    return output_data


def generate_with_transformers(
    model_path: str,
    prompts: list,
    output_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    seed: int = 42,
):
    """Generate responses using Transformers (fallback if vLLM not available)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Generate
    output_data = []
    for prompt in tqdm(prompts, desc="Generating"):
        # Format conversation (handle models with and without chat templates)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Use chat template for instruction-tuned models
            conversation = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use raw prompt for base models (e.g., GPT-2)
            conversation = prompt
        
        # Tokenize
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        output_data.append({
            'prompt': prompt,
            'format_prompt': conversation,
            'generated_text': generated_text,
        })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Generate responses for DistiLLM-2')
    parser.add_argument('--teacher-model', type=str, required=True,
                       help='Path to teacher model')
    parser.add_argument('--student-model', type=str, required=True,
                       help='Path to student model')
    parser.add_argument('--data-path', type=str, default='HuggingFaceH4/ultrachat_200k',
                       help='Path to prompt dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for generated data')
    parser.add_argument('--split', type=str, default='train_sft',
                       help='Dataset split to use')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to generate (None for all)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--use-vllm', action='store_true',
                       help='Use vLLM for generation (faster)')
    parser.add_argument('--split-type', type=str, default='train',
                       help='Split type: train or dev')
    
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading dataset from {args.data_path}...")
    
    # Load from JSONL file (use 'json' loader for .jsonl files)
    import os
    if os.path.isfile(args.data_path):
        dataset = load_dataset('json', data_files=args.data_path, split='train')
    else:
        # HuggingFace dataset name
        dataset = load_dataset(args.data_path, split=args.split)
    
    # Extract prompts (handle different formats)
    if 'messages' in dataset.column_names:
        prompts = [ex['messages'][0]['content'] for ex in dataset]
    elif 'prompt' in dataset.column_names:
        prompts = dataset['prompt']
    else:
        raise ValueError("Dataset must have 'messages' or 'prompt' column")
    
    # Limit samples if specified
    if args.num_samples:
        prompts = prompts[:args.num_samples]
    
    print(f"Generating responses for {len(prompts)} prompts...")
    
    # Create output paths
    teacher_output = os.path.join(args.output_dir, f'generated_{args.split_type}_teacher.jsonl')
    student_output = os.path.join(args.output_dir, f'generated_{args.split_type}_student.jsonl')
    
    # Choose generation method
    generate_fn = generate_with_vllm if (args.use_vllm and VLLM_AVAILABLE) else generate_with_transformers
    
    # Generate teacher responses
    print("\n" + "="*80)
    print("Generating TEACHER responses...")
    print("="*80)
    teacher_data = generate_fn(
        model_path=args.teacher_model,
        prompts=prompts,
        output_path=teacher_output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size if args.use_vllm else 1,
    )
    
    # Generate student responses
    print("\n" + "="*80)
    print("Generating STUDENT responses...")
    print("="*80)
    student_data = generate_fn(
        model_path=args.student_model,
        prompts=prompts,
        output_path=student_output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size if args.use_vllm else 1,
    )
    
    print("\n" + "="*80)
    print("Generation complete!")
    print(f"Teacher outputs: {teacher_output}")
    print(f"Student outputs: {student_output}")
    print("="*80)


if __name__ == "__main__":
    main()
