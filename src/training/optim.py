from __future__ import annotations

from torch.optim import AdamW


def build_optimizer(model, lr_proj: float, lr_backbone: float, weight_decay: float):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_proj, "weight_decay": weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    return AdamW(param_groups)
