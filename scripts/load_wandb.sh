#!/bin/bash
# Helper script to read wandb config from YAML file
# Usage: source scripts/load_wandb.sh

CONFIG_FILE="${BASE_PATH:-./}/wandb_config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    # Extract wandb key from YAML file - look for key: "..." pattern
    WANDB_KEY=$(grep "key:" "$CONFIG_FILE" | sed 's/.*key:[[:space:]]*"\([^"]*\)".*/\1/')
    
    if [ ! -z "$WANDB_KEY" ]; then
        export WANDB_KEY
        echo "✓ Wandb key loaded from $CONFIG_FILE: ${WANDB_KEY:0:10}..."
    else
        echo "⚠ Wandb key not found in $CONFIG_FILE"
        echo "  Looking for pattern: key: \"your-key-here\""
    fi
else
    echo "⚠ Wandb config file not found: $CONFIG_FILE"
fi
