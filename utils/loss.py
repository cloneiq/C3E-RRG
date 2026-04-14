"""
Language Model loss
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_im_loss(pred, imgs, mask, p=16, norm_pix_loss=False):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def compute_lm_loss(output, reports_ids, reports_masks):
    if isinstance(output, list):
        output = output[0]
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


def compute_recon_loss(pred, target, mask, mode='text'):
    if mode == 'text':
        loss = compute_lm_loss(pred, target, mask)
    elif mode == 'img':
        loss = compute_im_loss(pred, target, mask)
    else:
        raise ValueError
    return loss

# ===== DyMES: total loss aggregator =====
def dymes_total_loss(args, output, reports_ids, reports_masks, model=None, criterion=None):
    """
    统一计算总损失：CE + (lambda_cem)*cem + (lambda_back)*back_consistency + (lambda_cf)*cf + (lambda_fd_ortho)*fd_ortho
    - output: [log_probs, {}, meta] 由 DYMES(train) 返回
    - reports_ids/reports_masks: 来自 dataloader
    - model: 需访问 dyce 的缓存（X_hat/X_fin/X_CF/θ_norm）与四个 lambda 超参
    - criterion: 语言模型的 CE/NLL 损失
    """
    assert isinstance(output, (list, tuple)) and len(output) >= 1, "expect list output from model(train)"
    log_probs = output[0]
    meta = output[2] if (len(output) > 2 and isinstance(output[2], dict)) else {}

    # 1) CE / NLL
    if criterion is None:
        raise ValueError("criterion is required for CE/NLL.")
    loss_ce = criterion(output, reports_ids, reports_masks)

    # 2) CEM 对比损失
    loss_cem = torch.tensor(0.0, device=log_probs.device)
    if meta.get("cem_loss", None) is not None and getattr(model, "lambda_cem", 0.0) > 0:
        loss_cem = model.lambda_cem * meta["cem_loss"]

    # 3) DyCE 回向一致性（X_hat ~ X_fin）
    loss_back = torch.tensor(0.0, device=log_probs.device)
    if getattr(model, "lambda_back", 0.0) > 0:
        X_hat = getattr(model, "_X_hat", None)
        X_fin = getattr(model, "_X_path_final", None)
        if isinstance(X_hat, torch.Tensor) and isinstance(X_fin, torch.Tensor):
            loss_back = model.lambda_back * (X_hat - X_fin).pow(2).mean()

    # 4) 反事实正则（X_CF ~ θ_norm）
    loss_cf = torch.tensor(0.0, device=log_probs.device)
    if getattr(model, "lambda_cf", 0.0) > 0:
        X_CF  = getattr(model, "_X_CF", None)
        theta = getattr(model.dyce, "normal_template", None) if (model is not None and hasattr(model, "dyce")) else None
        if isinstance(X_CF, torch.Tensor) and isinstance(theta, torch.Tensor):
            loss_cf = model.lambda_cf * (X_CF - theta.expand_as(X_CF)).pow(2).mean()

    # 5) FD 正交（<X_path, X_noise> -> 0）
    loss_fd = torch.tensor(0.0, device=log_probs.device)
    if getattr(model, "lambda_fd_ortho", 0.0) > 0:
        xp = meta.get("fd_x_path", None)
        xn = meta.get("fd_x_noise", None)
        if isinstance(xp, torch.Tensor) and isinstance(xn, torch.Tensor):
            xp = F.normalize(xp, p=2, dim=-1)
            xn = F.normalize(xn, p=2, dim=-1)
            loss_fd = model.lambda_fd_ortho * (xp * xn).sum(-1).abs().mean()

    loss = loss_ce + loss_cem + loss_back + loss_cf + loss_fd

    terms = dict(
        loss=float(loss.item()),
        loss_ce=float(loss_ce.item()),
        loss_cem=float(loss_cem.item()) if torch.is_tensor(loss_cem) else float(loss_cem),
        loss_back=float(loss_back.item()) if torch.is_tensor(loss_back) else float(loss_back),
        loss_cf=float(loss_cf.item()) if torch.is_tensor(loss_cf) else float(loss_cf),
        loss_fd=float(loss_fd.item()) if torch.is_tensor(loss_fd) else float(loss_fd),
    )
    return loss, terms

