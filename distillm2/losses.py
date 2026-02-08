"""
DistillM-2 Loss Functions

Implementation of DistillM-2 distillation losses based on the paper:
"Towards Scalable Automated Alignment of LLMs: A Survey"

This module provides loss functions for knowledge distillation using 
the DistillM-2 methodology, which combines forward and reverse KL divergences
with adaptive mixing coefficients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def compute_position_kl(
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    loss_type: str = "distillm_v1",
    base_alpha_1: float = 0.1,
    base_alpha_2: float = 0.1,
    logp_logq: Optional[float] = None,
    logq_logp: Optional[float] = None,
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute position-wise KL divergence for DistillM-2.
    
    Args:
        student_logits: Logits from student model (batch_size, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        loss_mask: Mask for valid positions (batch_size, seq_len)
        loss_type: Type of loss ('distillm_v1' or 'distillm_v2')
        base_alpha_1: Base mixing coefficient for teacher-centric KL
        base_alpha_2: Base mixing coefficient for student-centric KL  
        logp_logq: Precomputed teacher/student log prob ratio (for adaptive alpha)
        logq_logp: Precomputed student/teacher log prob ratio (for adaptive alpha)
        global_step: Current training step (for gradual beta in v2)
        max_steps: Total training steps (for gradual beta in v2)
        
    Returns:
        Tuple of (tea_position_kl, ref_position_kl) where:
        - tea_position_kl: Teacher-centric KL = KL(teacher || mix_1)
        - ref_position_kl: Student-centric KL = KL(student || mix_2)
    """
    # Compute log probabilities
    student_vocab_logps = student_logits.log_softmax(-1)
    teacher_vocab_logps = teacher_logits.log_softmax(-1)
    
    # Get per-token log probs for the labels
    student_per_token_logps = torch.gather(
        student_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    teacher_per_token_logps = torch.gather(
        teacher_vocab_logps, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    
    if "distillm" in loss_type:
        # Compute adaptive alpha_1 for teacher KL (if enabled)
        try:
            assert loss_type == "distillm_v2" and logp_logq is not None
            anchor = (1 - base_alpha_1) * logp_logq
            logps_logqs = (
                (teacher_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            ).exp() - (
                (student_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            ).exp()
            alpha_1 = torch.clip(
                1 - anchor / (logps_logqs + 1e-5), 
                min=1e-2, 
                max=base_alpha_1
            ).unsqueeze(-1).unsqueeze(-1)
        except:
            alpha_1 = base_alpha_1
        
        # Compute teacher-centric KL with mixing
        try:
            if isinstance(alpha_1, torch.Tensor):
                log_alpha_1 = torch.log(alpha_1)
                log_one_minus_alpha_1 = torch.log(1 - alpha_1)
            else:
                log_alpha_1 = math.log(alpha_1)
                log_one_minus_alpha_1 = math.log(1 - alpha_1)
            
            mix_vocab_logps = torch.logsumexp(
                torch.stack([
                    log_alpha_1 + teacher_vocab_logps,
                    log_one_minus_alpha_1 + student_vocab_logps
                ], dim=0),
                dim=0
            )
            tea_pos_kl = (
                teacher_vocab_logps.exp() * (teacher_vocab_logps - mix_vocab_logps)
            ).sum(-1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if isinstance(alpha_1, torch.Tensor):
                log_alpha_1 = torch.log(alpha_1)
                log_one_minus_alpha_1 = torch.log(1 - alpha_1)
            else:
                log_alpha_1 = math.log(alpha_1)
                log_one_minus_alpha_1 = math.log(1 - alpha_1)
            
            mix_vocab_logps = torch.logsumexp(
                torch.stack([
                    log_alpha_1 + teacher_vocab_logps,
                    log_one_minus_alpha_1 + student_vocab_logps
                ], dim=0),
                dim=0
            )
            tea_pos_kl = (
                teacher_vocab_logps.exp() * (teacher_vocab_logps - mix_vocab_logps)
            ).sum(-1)
        del mix_vocab_logps
        
        # Compute adaptive alpha_2 for student KL (if enabled)
        try:
            assert loss_type == "distillm_v2" and logq_logp is not None
            anchor = (1 - base_alpha_2) * logq_logp
            logqs_logps = (
                (student_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            ).exp() - (
                (teacher_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            ).exp()
            alpha_2 = torch.clip(
                1 - anchor / (logqs_logps + 1e-5),
                min=1e-2,
                max=base_alpha_2
            ).unsqueeze(-1).unsqueeze(-1)
        except:
            alpha_2 = base_alpha_2
        
        # Compute student-centric KL with mixing
        try:
            if isinstance(alpha_2, torch.Tensor):
                log_alpha_2 = torch.log(alpha_2)
                log_one_minus_alpha_2 = torch.log(1 - alpha_2)
            else:
                log_alpha_2 = math.log(alpha_2)
                log_one_minus_alpha_2 = math.log(1 - alpha_2)
            
            mix_vocab_logps = torch.logsumexp(
                torch.stack([
                    log_one_minus_alpha_2 + teacher_vocab_logps,
                    log_alpha_2 + student_vocab_logps.detach()
                ], dim=0),
                dim=0
            )
            ref_pos_kl = (
                student_vocab_logps.exp() * (student_vocab_logps - mix_vocab_logps)
            ).sum(-1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if isinstance(alpha_2, torch.Tensor):
                log_alpha_2 = torch.log(alpha_2)
                log_one_minus_alpha_2 = torch.log(1 - alpha_2)
            else:
                log_alpha_2 = math.log(alpha_2)
                log_one_minus_alpha_2 = math.log(1 - alpha_2)
            
            mix_vocab_logps = torch.logsumexp(
                torch.stack([
                    log_one_minus_alpha_2 + teacher_vocab_logps,
                    log_alpha_2 + student_vocab_logps.detach()
                ], dim=0),
                dim=0
            )
            ref_pos_kl = (
                student_vocab_logps.exp() * (student_vocab_logps - mix_vocab_logps)
            ).sum(-1)
        del mix_vocab_logps
        del teacher_vocab_logps
        
    else:
        # Standard forward/reverse KL
        tea_pos_kl = (
            teacher_vocab_logps.exp() * (teacher_vocab_logps - student_vocab_logps)
        ).sum(-1)
        ref_pos_kl = (
            student_vocab_logps.exp() * (teacher_vocab_logps - student_vocab_logps)
        ).sum(-1)
    
    # Aggregate over sequence dimension
    tea_position_kl = (tea_pos_kl * loss_mask).sum(-1) / loss_mask.sum(-1)
    ref_position_kl = (ref_pos_kl * loss_mask).sum(-1) / loss_mask.sum(-1)
    student_all_logps = (student_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    teacher_all_logps = (teacher_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    
    # Clean up
    del teacher_per_token_logps
    del student_per_token_logps
    del tea_pos_kl
    del ref_pos_kl
    
    return student_all_logps, teacher_all_logps, tea_position_kl, ref_position_kl


def distillm_v1_loss(
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    loss_mask: torch.BoolTensor,
) -> torch.FloatTensor:
    """
    DistillM-v1 loss: Sum of student-centric KL divergences.
    
    Loss = KL(student || mix_2) + KL(student || mix_2)
    For v1, both chosen and rejected use ref_pos_kl (student-centric KL).
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits  
        labels: Target labels
        loss_mask: Mask for valid positions
        
    Returns:
        Scalar loss tensor
    """
    all_logps, tea_all_logps, tea_kl, ref_kl = compute_position_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        loss_mask=loss_mask,
        loss_type="distillm_v1",
    )
    
    # For distillm_v1: use ref_kl + ref_kl (both student-centric)
    loss = ref_kl + ref_kl
    return loss.mean()


def distillm_v2_loss(
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    loss_mask: torch.BoolTensor,
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
    gradual_beta: bool = False,
) -> torch.FloatTensor:
    """
    DistillM-v2 loss: Weighted sum with adaptive or gradual coefficients.
    
    Loss = (2-beta) * KL(teacher || mix_1) + beta * KL(student || mix_2)
    
    where beta can be:
    - Fixed at 1.0 (default)
    - Gradually increased from 1.0 to 1.5 during training (if gradual_beta=True)
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Target labels
        loss_mask: Mask for valid positions
        global_step: Current training step (for gradual beta)
        max_steps: Total training steps (for gradual beta)
        gradual_beta: Whether to use gradual beta schedule
        
    Returns:
        Scalar loss tensor
    """
    all_logps, tea_all_logps, tea_kl, ref_kl = compute_position_kl(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        loss_mask=loss_mask,
        loss_type="distillm_v2",
        global_step=global_step,
        max_steps=max_steps,
    )
    
    # Compute beta coefficient
    if gradual_beta and global_step is not None and max_steps is not None:
        beta = 1.0 + 0.5 * min(1.0, 2 * global_step / max_steps)
    else:
        beta = 1.0
    
    # For distillm_v2: tea_kl for chosen (teacher-centric), ref_kl for rejected (student-centric)
    loss = (2 - beta) * tea_kl + beta * ref_kl
    return loss.mean()


def get_distillm2_loss(
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    loss_type: str = "distillm_v2",
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
    gradual_beta: bool = False,
) -> torch.FloatTensor:
    """
    Main interface for computing DistillM-2 loss.
    
    Args:
        student_logits: Student model logits (batch_size, seq_len, vocab_size)
        teacher_logits: Teacher model logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len). If None, assumes all positions are valid.
        loss_type: Type of loss ('distillm_v1' or 'distillm_v2')
        global_step: Current training step
        max_steps: Total training steps
        gradual_beta: Whether to use gradual beta schedule (only for v2)
        
    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM: predict next token
    shift_logits_student = student_logits[..., :-1, :].contiguous()
    shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Create loss mask
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        loss_mask = (shift_labels != -100) & (shift_attention_mask == 1)
    else:
        loss_mask = (shift_labels != -100)
    
    # Set masked labels to 0 (dummy value, will be masked out)
    shift_labels = shift_labels.clone()
    shift_labels[~loss_mask] = 0
    
    if loss_type == "distillm_v1":
        return distillm_v1_loss(
            student_logits=shift_logits_student,
            teacher_logits=shift_logits_teacher,
            labels=shift_labels,
            loss_mask=loss_mask,
        )
    elif loss_type == "distillm_v2":
        return distillm_v2_loss(
            student_logits=shift_logits_student,
            teacher_logits=shift_logits_teacher,
            labels=shift_labels,
            loss_mask=loss_mask,
            global_step=global_step,
            max_steps=max_steps,
            gradual_beta=gradual_beta,
        )
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Should be 'distillm_v1' or 'distillm_v2'"
        )


def get_distillm2_loss_split(
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    loss_type: str = "distillm_v2",
    logp_logq: float | None = None,
    logq_logp: float | None = None,
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute DistillM-2 position-wise KL for splitting into chosen/rejected.
    
    This function returns BOTH tea_pos_kl and ref_pos_kl without combining them,
    allowing the caller to split and combine with gradual beta externally.
    
    Args:
        student_logits: Student model logits (batch_size, seq_len, vocab_size)
        teacher_logits: Teacher model logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)
        loss_type: Type of loss ('distillm_v1' or 'distillm_v2')
        global_step: Current training step (for adaptive alpha)
        max_steps: Total training steps (for adaptive alpha)
        
    Returns:
        Tuple of (tea_position_kl, ref_position_kl) tensors, each of shape (batch_size,)
    """
    # Shift for causal LM: predict next token
    shift_logits_student = student_logits[..., :-1, :].contiguous()
    shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Create loss mask
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        loss_mask = (shift_labels != -100) & (shift_attention_mask == 1)
    else:
        loss_mask = (shift_labels != -100)
    
    # Set masked labels to 0 (dummy value, will be masked out)
    shift_labels = shift_labels.clone()
    shift_labels[~loss_mask] = 0
    
    # Compute position-wise KL and return both terms
    all_logps, tea_all_logps, tea_pos_kl, ref_pos_kl = compute_position_kl(
        student_logits=shift_logits_student,
        teacher_logits=shift_logits_teacher,
        labels=shift_labels,
        loss_mask=loss_mask,
        loss_type=loss_type,
        logp_logq=logp_logq,
        logq_logp=logq_logp,
        global_step=global_step,
        max_steps=max_steps,
    )
    
    return all_logps, tea_all_logps, tea_pos_kl, ref_pos_kl
