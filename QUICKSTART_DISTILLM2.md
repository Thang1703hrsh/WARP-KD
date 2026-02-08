# Quick Start - DistiLLM-2 Integration

## Integration Complete ✅

DistiLLM-2 is now fully integrated into `finetune.py` following the existing codebase architecture.

## How to Run

### GPT-2 (1.5B → 0.1B)

**Standalone DistiLLM-2** (uses OpenWebText):
```bash
bash scripts/gpt2/distillm2/distillm2_v2_base.sh
```

**DistiLLM-2 + Contra-KD** (uses OpenWebText):
```bash
bash scripts/gpt2/distillm2/contra_0.1B_1.5B.sh
```

**Full pipeline** (train + eval):
```bash
bash runs/gpt2/distillm2.sh
```

### OpenLLaMA-2 (7B → 3B with LoRA)

**Standalone DistiLLM-2** (uses OpenWebText):
```bash
bash scripts/openllama2/distillm2/distillm2_3B_7B_lora.sh
```

**DistiLLM-2 + Contra-KD** (uses OpenWebText):
```bash
bash scripts/openllama2/distillm2/contra_3B_7B_lora.sh
```

### LLaMA-2 (13B → 7B with LoRA)

**Standalone DistiLLM-2** (uses OpenWebText):
```bash
bash scripts/llama2/distillm2/distillm2_7B_13B_lora.sh
```

**DistiLLM-2 + Contra-KD** (uses OpenWebText):
```bash
bash scripts/llama2/distillm2/contra_7B_13B_lora.sh
```

## Key Changes

### finetune.py
1. **Arrow Dataset Support**: Automatically detects `distillm2` type and loads Arrow datasets
2. **Custom Dataset Class**: `DistiLLM2Dataset` wrapper with proper `collate()` method
3. **Loss Integration**: `get_distil_loss()` now handles `--type distillm2-v2`
4. **Global Step Tracking**: Added for gradual beta scheduling

### Data Format
- **Location**: `data/distillm2/{model}/formatted/`
- **Format**: HuggingFace Arrow datasets (dataset_dict.json + train/test folders)
- **Fields**: `prompt`, `chosen`, `rejected`

### Script Structure
```
runs/{model}/distillm2.sh               # Orchestration
└── scripts/{model}/distillm2/train.sh  # Training call to finetune.py
```

## Configuration

### Essential Arguments
```bash
--do-train                           # Enable training mode
--type distillm2-v2                  # Use DistiLLM-2 loss
--data-dir data/distillm2/gpt2/formatted  # Arrow dataset path
--distillm2-loss-type distillm_v2    # Loss variant (v1 or v2)
--base-alpha-1 0.1                   # Alpha parameter 1
--base-alpha-2 0.1                   # Alpha parameter 2
--lr 5e-4                            # Learning rate
--lr-decay-style cosine              # LR scheduler
--warmup-iters 100                   # Warmup steps
--epochs 20                          # Training epochs
--batch-size 16                      # Batch size
--kd-ratio 0.5                       # KD ratio for LM data mixing
```

## Verify Installation

### Check Data
```bash
ls data/distillm2/gpt2/formatted/
# Should show: dataset_dict.json  train/  test/
```

### Test Loading
```python
from datasets import load_from_disk
ds = load_from_disk("data/distillm2/gpt2/formatted")
print(ds)  # Should show train/test splits
print(ds['train'][0].keys())  # Should show: prompt, chosen, rejected
```

### Dry Run (10 iterations)
```bash
deepspeed --num_gpus=1 finetune.py \
    --do-train \
    --type distillm2-v2 \
    --model-path results/gpt2/train/init/gpt2-base \
    --teacher-model-path results/gpt2/train/sft/gpt2-xlarge \
    --data-dir data/distillm2/gpt2/formatted \
    --batch-size 4 \
    --max-length 512 \
    --lr 5e-4 \
    --epochs 1 \
    --save-interval 100000
```

## Performance Notes

- ✅ Gradient accumulation properly implemented
- ✅ LR scheduler steps correctly  
- ✅ Global step tracking for gradual beta
- ✅ Identical loss computation from distillm2/losses.py
- ✅ Same tokenization logic as distillm-2-master

**Expected**: Performance should match distillm-2-master when using same data/hyperparameters.

## Cleanup

After verifying integration works, you can delete:
- `train_distillm2.py` (functionality moved to finetune.py)

## Full Documentation

See `DISTILLM2_INTEGRATION.md` for complete details on:
- Data preparation
- Advanced configuration
- Troubleshooting
- Performance tuning
