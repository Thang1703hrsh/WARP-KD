"""
Reformat teacher-student paired data for DistiLLM-2 training.

This script takes the separately generated teacher and student responses
and combines them into the paired format needed for DistiLLM-2.
"""

import os
import json
import argparse
from datasets import load_dataset, DatasetDict


def reformat_data(teacher_file: str, student_file: str, output_dir: str):
    """
    Reformat teacher and student outputs into DistiLLM-2 format.
    
    Expected input format (JSONL):
    - prompt: The input prompt
    - generated_text: The model's response
    
    Output format (JSON):
    - prompt: The input prompt
    - chosen: Teacher's response (higher quality)
    - rejected: Student's response (lower quality)
    """
    print(f"Loading teacher data from {teacher_file}...")
    with open(teacher_file, 'r') as f:
        teacher_data = json.load(f)
    
    print(f"Loading student data from {student_file}...")
    with open(student_file, 'r') as f:
        student_data = json.load(f)
    
    # Verify same prompts
    assert len(teacher_data) == len(student_data), \
        f"Mismatch in data length: teacher={len(teacher_data)}, student={len(student_data)}"
    
    # Create paired samples
    samples = []
    for teacher_ex, student_ex in zip(teacher_data, student_data):
        # Verify prompts match
        assert teacher_ex['prompt'] == student_ex['prompt'], \
            f"Prompt mismatch: {teacher_ex['prompt']} != {student_ex['prompt']}"
        
        samples.append({
            'prompt': teacher_ex['prompt'],
            'chosen': teacher_ex['generated_text'],  # Teacher = chosen (better)
            'rejected': student_ex['generated_text'],  # Student = rejected (worse)
        })
    
    print(f"Created {len(samples)} paired samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Reformat DistiLLM-2 data')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing raw generated outputs')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for reformatted data')
    
    args = parser.parse_args()
    
    # Process train split
    print("\n" + "="*80)
    print("Processing TRAIN split...")
    print("="*80)
    teacher_train = os.path.join(args.input_dir, 'generated_train_teacher.jsonl')
    student_train = os.path.join(args.input_dir, 'generated_train_student.jsonl')
    
    if not os.path.exists(teacher_train) or not os.path.exists(student_train):
        raise FileNotFoundError(f"Missing files:\n  {teacher_train}\n  {student_train}")
    
    train_samples = reformat_data(teacher_train, student_train, args.output_dir)
    
    # Save as JSON files
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, 'train.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    print(f"\nSaved train data to {train_path}")
    
    # Try to process dev/test split (if available, otherwise use subset of train)
    print("\n" + "="*80)
    print("Processing DEV/TEST split...")
    print("="*80)
    
    teacher_dev = os.path.join(args.input_dir, 'generated_dev_teacher.jsonl')
    student_dev = os.path.join(args.input_dir, 'generated_dev_student.jsonl')
    
    # Check if dev files exist
    if os.path.exists(teacher_dev) and os.path.exists(student_dev):
        print("Found dev files, using them for test split...")
        test_samples = reformat_data(teacher_dev, student_dev, args.output_dir)
    else:
        print("No dev files found, using first 500 train samples as test split...")
        test_samples = train_samples[:500]  # Use first 500 as test split
    
    dev_path = os.path.join(args.output_dir, 'dev.json')
    
    with open(dev_path, 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"Saved dev/test data to {dev_path}")
    
    # Create and save as Arrow format (for faster loading)
    print("\nCreating Arrow dataset...")
    dataset = DatasetDict({
        'train': load_dataset('json', data_files=train_path, split='train'),
        'test': load_dataset('json', data_files=dev_path, split='train'),
    })
    
    dataset.save_to_disk(args.output_dir)
    print(f"Saved Arrow dataset to {args.output_dir}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Sample examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    for i, ex in enumerate(train_samples[:2]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {ex['prompt'][:100]}...")
        print(f"Chosen (Teacher): {ex['chosen'][:100]}...")
        print(f"Rejected (Student): {ex['rejected'][:100]}...")


if __name__ == "__main__":
    main()
