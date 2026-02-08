import os
import importlib.util
import torch
import torch.nn.functional as F

_DTW_MODULE = None


def _get_soft_dtw_class():
    global _DTW_MODULE
    if _DTW_MODULE is None:
        dtw_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "DTW-KD", "soft_dtw_cuda.py")
        )
        if not os.path.exists(dtw_path):
            raise FileNotFoundError(f"SoftDTW module not found at {dtw_path}")
        spec = importlib.util.spec_from_file_location("soft_dtw_cuda", dtw_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DTW_MODULE = module
    return _DTW_MODULE.SoftDTW


def _resolve_base_model(model):
    return model.module if hasattr(model, "module") else model


def _get_embed_tokens(model):
    base_model = _resolve_base_model(model)
    if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
        return base_model.model.embed_tokens
    if hasattr(base_model, "model") and hasattr(base_model.model, "model") and hasattr(base_model.model.model, "embed_tokens"):
        return base_model.model.model.embed_tokens
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "wte"):
        return base_model.transformer.wte
    raise NotImplementedError("Unsupported model architecture for embed_tokens")


def _get_projected(x, projector):
    if projector is None:
        return x
    return projector(x)


def _calculate_alignment_loss(student_embs, teacher_embs, student_mask, teacher_mask, soft_dtw, args):
    batch_size = student_embs.size(0)
    total_loss = torch.tensor(0.0, device=student_embs.device, requires_grad=True)
    non_empty_pairs = 0

    for i in range(batch_size):
        s_len = int(student_mask[i].sum().item())
        t_len = int(teacher_mask[i].sum().item())
        if s_len == 0 or t_len == 0:
            continue

        non_empty_pairs += 1
        s_seq = student_embs[i, :s_len, :]
        t_seq = teacher_embs[i, :t_len, :]

        c_stu_tea = 1.0 - torch.cosine_similarity(
            s_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
        )
        c_stu_stu = 1.0 - torch.cosine_similarity(
            s_seq.unsqueeze(1), s_seq.unsqueeze(0), dim=-1
        )
        c_tea_tea = 1.0 - torch.cosine_similarity(
            t_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
        )

        if getattr(args, "dtw_band_source", "none") == "sdtw" and getattr(args, "dtw_band_width", 0) > 0:
            _, align = soft_dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0), return_alignment=True)
            align = align[0]
            eps = 1e-9
            align_clamped = (align + eps) / (align.sum(dim=-1, keepdim=True) + eps)
            row_entropy = -(align_clamped * torch.log(align_clamped)).sum(dim=-1)

            lin_center = torch.arange(s_len, device=align.device, dtype=torch.float32) * (float(t_len) / float(s_len))
            soft_center = (align_clamped * torch.arange(t_len, device=align.device).view(1, -1)).sum(dim=-1)
            alpha = float(getattr(args, "dtw_band_center_blend", 0.7))
            centers = alpha * soft_center + (1.0 - alpha) * lin_center

            base_w = float(getattr(args, "dtw_band_width", 0))
            width = base_w + float(getattr(args, "dtw_band_entropy_coef", 2.0)) * row_entropy

            j = torch.arange(t_len, device=align.device).view(1, -1).float()
            dist = (j - centers.view(-1, 1)).abs()
            band = dist <= width.view(-1, 1)

            if getattr(args, "dtw_band_warmup_steps", 0) > 0:
                current_step = getattr(args, "current_global_step", 0)
                pen_scale = min(1.0, float(current_step + 1) / float(args.dtw_band_warmup_steps))
            else:
                pen_scale = 1.0
            penalty = float(getattr(args, "dtw_band_penalty", 1.0)) * pen_scale
            c_stu_tea = c_stu_tea + (~band).float() * penalty

        s2t = soft_dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
        s2s = soft_dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
        t2t = soft_dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))

        pair_loss = s2t - 0.5 * (s2s + t2t)
        total_loss = total_loss + pair_loss.squeeze()

    if non_empty_pairs == 0:
        return torch.tensor(0.0, device=student_embs.device, requires_grad=True)

    return total_loss


def dtw_distillation_loss(
    student_hidden_states,
    teacher_hidden_states,
    labels,
    student_model,
    teacher_model,
    projector,
    args,
):
    pad_mask = labels.ne(-100)
    if pad_mask.sum() == 0:
        return torch.tensor(0.0, device=labels.device, dtype=torch.float32)

    SoftDTW = _get_soft_dtw_class()

    dtw_gamma_start = getattr(args, "dtw_gamma_start", getattr(args, "dtw_gamma", 2.0))
    dtw_gamma_end = getattr(args, "dtw_gamma_end", 0.8)
    dtw_gamma_steps = getattr(args, "dtw_gamma_steps", 0)
    current_step = getattr(args, "current_global_step", 0)

    if dtw_gamma_steps and dtw_gamma_steps > 0:
        progress = min(1.0, float(current_step + 1) / float(dtw_gamma_steps))
        current_gamma = dtw_gamma_start + (dtw_gamma_end - dtw_gamma_start) * progress
    else:
        current_gamma = dtw_gamma_start

    soft_dtw = SoftDTW(
        use_cuda=labels.is_cuda,
        gamma=current_gamma,
        normalize=False,
        bandwidth=getattr(args, "dtw_band_width", None),
        alignment_postprocess="row",
    )

    student_hiddens = student_hidden_states[-1]
    teacher_hiddens = teacher_hidden_states[-1].detach()

    if projector is None and student_hiddens.size(-1) != teacher_hiddens.size(-1):
        raise ValueError(
            "DTW loss requires a projector when student/teacher hidden sizes differ: "
            f"{student_hiddens.size(-1)} vs {teacher_hiddens.size(-1)}"
        )

    projected_student_hiddens = _get_projected(student_hiddens, projector)
    projected_teacher_hiddens = teacher_hiddens

    student_embed_tokens = _get_embed_tokens(student_model)
    teacher_embed_tokens = _get_embed_tokens(teacher_model)

    formal_target = torch.where(pad_mask, labels, torch.zeros_like(labels))
    student_target_embeds = student_embed_tokens(formal_target)
    teacher_target_embeds = teacher_embed_tokens(formal_target).detach()

    projected_student_embeds = _get_projected(student_target_embeds, projector)
    projected_teacher_embeds = teacher_target_embeds

    loss_embed = _calculate_alignment_loss(
        projected_student_embeds,
        projected_teacher_embeds,
        pad_mask,
        pad_mask,
        soft_dtw,
        args,
    )

    loss_hidden = _calculate_alignment_loss(
        projected_student_hiddens,
        projected_teacher_hiddens,
        pad_mask,
        pad_mask,
        soft_dtw,
        args,
    )

    total_dtw_loss = loss_hidden + loss_embed

    if getattr(args, "dtw_warmup_steps", 0) > 0:
        warmup_scale = min(1.0, float(current_step + 1) / float(args.dtw_warmup_steps))
    else:
        warmup_scale = 1.0

    dtw_rate = getattr(args, "dtw_rate", 1.0)
    return total_dtw_loss * warmup_scale * dtw_rate

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def l2_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes normalized L2 loss only for valid (non-padded) tokens."""
    # Compute mean squared error per token (averaged over vocab dimension)
    mse_per_token = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [B, L]
    # Apply mask and average over valid tokens
    masked_losses = mse_per_token * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return masked_losses.sum() / valid_tokens
    else:
        return torch.tensor(0.0, device=pred.device)


def cosine_similarity_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes masked cosine similarity loss."""
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(pred.float(), target.float(), dim=-1)  # [B, L]
    # Apply mask and compute loss for valid tokens only
    masked_cos_sim = cos_sim * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return (1 - masked_cos_sim.sum() / valid_tokens)
    else:
        return torch.tensor(0.0, device=pred.device)


def hybrid_loss_masked(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    cosine_weight: float = 0.6, 
    l2_weight: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid loss combining cosine similarity and L2 loss.
    
    Args:
        pred: Predicted logits [B, L, V]
        target: Target logits [B, L, V] 
        mask: Attention mask [B, L]
        cosine_weight: Weight for cosine similarity loss (emphasizes direction)
        l2_weight: Weight for L2 loss (emphasizes magnitude)
    
    Returns:
        Combined loss value, cosine loss, l2 loss
    """
    # Cosine similarity loss (for directional alignment)
    cosine_loss = cosine_similarity_loss_masked(pred, target, mask)
    
    # L2 loss (for magnitude preservation)
    l2_loss = l2_loss_masked(pred, target, mask)
    
    # Combine with weights
    hybrid_loss = cosine_weight * cosine_loss + l2_weight * l2_loss
    
    return hybrid_loss, cosine_loss, l2_loss


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes 1 - mean(cosine_similarity(a, b))."""
    return (1 - F.cosine_similarity(a.float(), b.float(), dim=-1)).mean()


def velocity_field_loss(
    student_hiddens, 
    teacher_hiddens,
    velocity_field, 
    projector,
    teacher_schedule,
    student_schedule,
    attention_mask,
    device=0
):
    """
    Compute FRFD velocity field loss for rectified flow distillation.
    """
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    # Sample time t once per batch
    batch_size = student_hiddens[0].size(0)
    t = torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32)
    num_distill_layers = len(teacher_schedule)
    
    # Loop over all distillation layers
    for j, (teacher_layer_idx, student_layer_idx) in enumerate(zip(teacher_schedule, student_schedule)):
        # Get hidden states for the current layer pair
        y_S = student_hiddens[student_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        y_T = teacher_hiddens[teacher_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        
        # Project student features to teacher's dimension
        y_S = projector(y_S)

        # Create interpolated features Y_t
        Y_t = (1 - t) * y_S + t * y_T
        
        # Compute target velocity
        target_velocity = y_T - y_S
        
        # Predict velocity using the velocity field model
        layer_indices = torch.tensor([j] * y_S.size(0), device=device, dtype=torch.long)
        predicted_velocity = velocity_field(Y_t, t.squeeze(1).squeeze(1), layer_indices)
        
        # Accumulate the MSE loss for this layer
        loss_per_token = F.mse_loss(predicted_velocity, target_velocity, reduction='none').mean(dim=-1)
        loss_per_token *= attention_mask
        loss = loss_per_token.sum() / attention_mask.sum()
        total_loss += loss / num_distill_layers
    
    return total_loss


def frfd_distillation_loss(
    student_hiddens,
    velocity_field,
    projector,
    student_schedule,
    attention_mask,
    num_distill_layers,
    device=0
):
    """
    Compute FRFD rectified flow distillation loss.
    Exactly matches the original FRFD stage2 implementation.
    """
    # Calculate Rectified Flow Distillation Loss
    loss_rfd = 0
    # delta_t = 1.0 / (num_distill_layers - 1)
    
    if attention_mask.sum() > 0:
        for j in range(num_distill_layers):
            h_S_current = student_hiddens[student_schedule[j]].to(device=f"cuda:{device}", dtype=torch.float32)
            # h_S_next = student_hiddens[student_schedule[j+1]].to(device=f"cuda:{device}", dtype=torch.float32)
            
            actual_y_next = projector(h_S_current)
            with torch.no_grad():
                y_S_j = projector(h_S_current.detach())
                B, L, V = y_S_j.shape
                
                # Get ideal update from velocity field
                t = torch.full((B,), 0, device=device, dtype=torch.float32)
                layer_indices = torch.full((B,), j, device=device, dtype=torch.long)
                ideal_update = velocity_field(y_S_j, t, layer_indices)
                
                # Target for next layer: Euler step from current layer
                target_y_next = y_S_j + ideal_update #* delta_t
            
            layer_loss = cosine_similarity_loss_masked(actual_y_next, target_y_next, attention_mask)
            loss_rfd += layer_loss / (num_distill_layers)
            
    else:
        loss_rfd = torch.tensor(0.0, device=device)
    
    return loss_rfd

def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature = 2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

    loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss

def get_fdd_loss(t_hiddens, s_hiddens, mask, teacher, student, teacher_schedule, student_schedule):
    i = 0
    traj_loss, der_loss = 0.0, 0.0
    pre_s_hidden_logs, pre_t_hidden_logs = None, None
    # mask = (no_model_batch["label"] != -100).int()

    for s_idx, t_idx in zip(student_schedule, teacher_schedule):
        s_hidden = s_hiddens[s_idx]
        t_hidden = t_hiddens[t_idx]
        # if args.model_type == 'opt':
        #     s_decoder_proj = student.module.model.model.decoder.project_out
        #     if s_decoder_proj is not None:
        #         s_hidden = s_decoder_proj(s_hidden)

        #     t_decoder_proj = teacher.model.decoder.project_out
        #     if t_decoder_proj is not None:
        #         t_hidden = t_decoder_proj(t_hidden)

        s_hidden_logits = student.module.lm_head(s_hidden)
        t_hidden_logits = teacher.lm_head(t_hidden)
        # traj_loss += forward_kl(s_hidden_logits, t_hidden_logits, no_model_batch)
        traj_loss += soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
        t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

        if i > 0:
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
            cos_sim_loss = 1 - cos_sim
            cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()

            der_loss +=  cos_sim_loss

        pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs

        i += 1

    return traj_loss / i +  der_loss / (i - 1)