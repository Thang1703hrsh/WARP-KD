#!/bin/bash

# Reformat DistiLLM-2 Data Script
# This script reformats the separately generated teacher and student responses
# into the paired format needed for DistiLLM-2 training

# Configuration
TEACHER_DIR="data/distillm2_paired"
STUDENT_DIR="data/distillm2_paired"
OUTPUT_DIR="data/distillm2_formatted"

echo "================================================"
echo "Reformatting DistiLLM-2 data..."
echo "================================================"

python reformat_distillm2_data.py \
    --teacher-dir ${TEACHER_DIR} \
    --student-dir ${STUDENT_DIR} \
    --output-dir ${OUTPUT_DIR}

echo "================================================"
echo "Reformatting complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "================================================"
