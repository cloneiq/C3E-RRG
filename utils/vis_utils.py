"""
Visualization utilities for DyMES.
- export_epoch_visuals(trainer, epoch): CHI/文本/回向一致性等可视化导出
- export_x_token_attn_maps(trainer, epoch): 编码器中 X_path-token 对 patch 的注意力热力图（门控对比）
"""

import os
import numpy as np
import torch
from PIL import Image
# OpenCV is used for resizing attention heatmaps; if not available, you can switch to PIL.Image.resize
import cv2
from utils.monitor import make_overlay, to_rgb_u8_from_tensor, upsample_to, robust_minmax
from pathlib import Path

@torch.no_grad()
def export_epoch_visuals(trainer, epoch: int):
    """
    导出：
    - CHI 热图叠加可视化（每个 epoch 从 val batch 采样最多 2 张）
    - 回向一致性（X_hat vs X_path_final）与 CHI 统计写入 txt
    - 当前样本预测文本（pred.txt）
    注意：trainer 需包含 .model / .val_dataloader / .args / .tokenizer 等属性（与 FinetuneTrainer 一致）。
    """
    vis_dir = os.path.join(trainer.args["result_dir"], "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # 取一个 val batch；不同代码里可能叫 valid_data_loader / val_dataloader
    loader = getattr(trainer, "val_dataloader", None)
    if loader is None:
        loader = getattr(trainer, "valid_data_loader", None)
    if loader is None:
        return

    try:
        batch_vis = next(iter(loader))
    except StopIteration:
        return

    # 解包（你的 loader 返回四元组）
    images_id, images, reports_ids, reports_masks = batch_vis
    images = images.cuda(non_blocking=True)
    reports_ids = reports_ids.cuda(non_blocking=True)

    # 先跑一次前向，写入中间量缓存（不回传梯度）
    trainer.model.eval()
    with torch.enable_grad():                   # ⬅️ 局部打开梯度
        _ = trainer.model(images, reports_ids, mode='train')


    # 安全地拿到 CHI（优先高分辨率）
    H_chi = getattr(trainer.model, "_last_H_chi_hr", None)  # [B,H,W]
    if H_chi is None:
        H_chi = getattr(trainer.model, "_last_H_chi", None)  # [B,32,32] 或 [B,1,32,32]
        if H_chi is not None and H_chi.dim() == 4 and H_chi.size(1) == 1:
            H_chi = H_chi.squeeze(1)

    # 计算可视化的最大样本数（避免越界）
    B_img = images.size(0)
    B_chi = H_chi.size(0) if (H_chi is not None and hasattr(H_chi, "size")) else B_img
    B_use = min(B_img, B_chi)
    if B_use <= 0:
        return
    n_show = min(2, B_use)  # 最多显示2张，但不能超过 batch 尺寸

    for i in range(n_show):
        # 取原图（双视角取第一视角）
        if images.dim() == 5:       # [B,2,3,H,W]
            img_t = images[i, 0]
        else:                       # [B,3,H,W]
            img_t = images[i]
        rgb = to_rgb_u8_from_tensor(img_t)

        # 取对应的 CHI 热图；若没有 HR，就回退 LR 并上采样到输入大小
        hmap = None
        if H_chi is not None:
            hmap_t = H_chi[i].detach().float().cpu()
            if hmap_t.dim() == 3 and hmap_t.size(0) == 1:
                hmap_t = hmap_t.squeeze(0)
            hmap = hmap_t.numpy()
            if hmap.shape[0] != rgb.shape[0] or hmap.shape[1] != rgb.shape[1]:
                hmap = upsample_to(hmap, rgb.shape[0], rgb.shape[1])

        # 叠加并保存（若没有 CHI，就跳过这张图）
        if hmap is not None:
            over = make_overlay(rgb, hmap, alpha=0.45)
            # 可选：导出大图更清晰
            scale = int(trainer.args.get("vis_upscale", 2))
            if scale > 1:
                over = Image.fromarray(over).resize(
                    (over.shape[1]*scale, over.shape[0]*scale), resample=Image.NEAREST
                )
                over = np.array(over)
            Image.fromarray(over).save(os.path.join(vis_dir, f"epoch_{epoch:03d}_sample{i}_chi.png"))

        # 回向一致性与 CHI 统计
        back_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_sample{i}_backcons.txt")
        chi_stats = getattr(trainer.model, "_last_chi_stats", None)

        X_hat = getattr(trainer.model, "_X_hat", None)
        X_path_final = getattr(trainer.model, "_X_path_final", None)
        if X_hat is not None and X_path_final is not None and i < X_hat.size(0) and i < X_path_final.size(0):
            err = (X_hat[i] - X_path_final[i]).detach().float()
            err_val = float(err.pow(2).mean().sqrt().item())
            with open(back_path, "w") as f:
                f.write(f"||X_hat - X_path||_2 = {err_val:.6f}\n")
            # 追加 CHI 统计
            if chi_stats is not None:
                with open(back_path, "a") as f:
                    f.write(f"CHI stats: min={chi_stats[0]:.6f}, max={chi_stats[1]:.6f}, std={chi_stats[2]:.6f}\n")
        else:
            # 即使没有回向一致性张量，也把 CHI 统计单独写出来，便于排查
            if chi_stats is not None:
                with open(back_path, "w") as f:
                    f.write(f"CHI stats: min={chi_stats[0]:.6f}, max={chi_stats[1]:.6f}, std={chi_stats[2]:.6f}\n")

        # 当前样本预测文本
        _ = trainer.model(images[i:i+1], mode='sample')
        pred_txt = getattr(trainer.model, "_last_report_text", "")
        with open(os.path.join(vis_dir, f"epoch_{epoch:03d}_sample{i}_pred.txt"), "w", encoding="utf-8") as f:
            f.write(pred_txt)


@torch.no_grad()
def export_x_token_attn_maps(trainer, epoch: int):
    """
    可视化：编码器最后一层里，X_path token（病理token）对各视觉patch的注意力权重热力图；
    并对比 x_token_gate 打开/关闭（1.0 vs 0.0）的差异。
    保存两张叠加在原图上的可视化： *_gate_on.png / *_gate_off.png
    """
    vis_dir = os.path.join(trainer.args["result_dir"], "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # 取一个 val batch
    loader = getattr(trainer, "val_dataloader", None)
    if loader is None:
        return
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    images_id, images, reports_ids, reports_masks = batch
    images = images.cuda(non_blocking=True)
    reports_ids = reports_ids.cuda(non_blocking=True)

    # 选择第一个样本做可视化
    bidx = 0
    # 取单视角图像（如果是双视角，就取第一视角做叠加）
    if images.dim() == 5:       # [B,2,3,H,W]
        img_t = images[bidx, 0]
    else:
        img_t = images[bidx]
    rgb = to_rgb_u8_from_tensor(img_t)

    def run_and_get_attn(xgate: float):
        """
        x_token_gate = xgate 运行一次前向，读取 Encoder 最后一层的 self-attn。
        返回：head-avg 后的 x_token -> all 的注意力向量 attn_vec [L]
        """
        m = trainer.model
        # 暂存原 gate
        gate_backup = getattr(m, "x_token_gate", None)
        if gate_backup is None:
            # 未启用病理 token，直接返回 None
            return None

        # 设置门控（注意：替换为新的 Parameter 以保持一致性）
        m.x_token_gate = torch.nn.Parameter(torch.tensor(float(xgate), device=images.device))

        # 走一遍前向（train 模式能最稳定触发 Encoder）
        _ = m(images, reports_ids, mode='train')

        # 取最后一层 Encoder 的注意力（MultiHeadedAttention.attn）
        try:
            enc = m.encoder
            last_layer = enc.layers[-1]
            attn = last_layer.attn.attn  # [B, H, L, L] 需 Encoder 层在 forward 里缓存 attn
            if attn is None:
                return None
            attn = attn.detach().float()  # 不留梯度
        except Exception:
            return None
        finally:
            # 复原 gate
            try:
                m.x_token_gate = gate_backup
            except Exception:
                pass

        # 取第 bidx 个样本，head 平均
        a = attn[bidx].mean(dim=0)        # [L, L]
        L = a.size(0)
        # x_token 在序列最后（CLS 在最前）
        x_idx = L - 1
        attn_vec = a[x_idx]               # [L]
        return attn_vec

    # 两次：gate on/off
    att_on  = run_and_get_attn(1.0)
    att_off = run_and_get_attn(0.0)

    def vector_to_heatmap(attn_vec):
        """
        把 [L] 的注意力向量转为 patch 热图（可能是单视角14x14，也可能是双视角 2*14x14 横向拼接）
        """
        if attn_vec is None: 
            return None
        attn_vec = attn_vec.detach().cpu().numpy() if torch.is_tensor(attn_vec) else np.asarray(attn_vec)
        # 去掉 CLS 与 x_token 自身，只保留 patch tokens
        L = attn_vec.shape[0]
        if L < 3: 
            return None
        patch_vec = attn_vec[1: L-1]                          # [Lv]
        Lv = patch_vec.shape[0]
        # 估算视角数（每视角 14x14=196 个 patch）
        nviews = int(round(Lv / 196.0))
        if nviews < 1: 
            nviews = 1
        # 如果不是 196 的整倍数，则回退为 sqrt 近似
        if Lv % 196 != 0:
            side = int(round(np.sqrt(Lv)))
            side = max(1, side)
            grid = patch_vec[:side*side].reshape(side, side)
            return grid
        # 否则视角分组 → 横向拼接（IU是2视角）
        grids = []
        for i in range(nviews):
            seg = patch_vec[i*196:(i+1)*196]
            g = seg.reshape(14,14)
            grids.append(g)
        grid = np.concatenate(grids, axis=1) if len(grids) > 1 else grids[0]
        return grid

    def save_overlay(grid, suffix: str):
        if grid is None: 
            return
        g = grid.astype(np.float32)
        # 归一化
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        # 上采样到图像大小（保持与原图对齐；如果是两视角，grid 宽度已横向拼接）
        Ht, Wt = rgb.shape[0], rgb.shape[1]
        g_up = cv2.resize(g, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        # 叠加可视化（用项目现有的 overlay）
        over = make_overlay(rgb, g_up, alpha=0.45)
        Image.fromarray(over).save(os.path.join(
            vis_dir, f"epoch_{epoch:03d}_xattn_{suffix}.png"))

    grid_on  = vector_to_heatmap(att_on)
    grid_off = vector_to_heatmap(att_off)
    save_overlay(grid_on,  "gate_on")
    save_overlay(grid_off, "gate_off")

@torch.no_grad()
def _pick_vis_batch(trainer, max_samples=2):
    """从 valid_loader 取一小批样本用于可视化；找不到就退回 train_loader。"""
    loader = getattr(trainer, "valid_loader", None) or getattr(trainer, "train_loader", None)
    batch = next(iter(loader))
    images = batch["images"][:max_samples].to(next(trainer.model.parameters()).device)
    targets = batch.get("reports_ids", None)
    if targets is not None:
        targets = targets[:max_samples].to(images.device)
    return images, targets

def export_grad_visuals(trainer, epoch: int, max_samples: int = 2):
    """
    梯度型可视化（例如 S_g_CEM 热图）。默认关闭，只在需要时由 Trainer 调用。
    """
    model = trainer.model
    model.eval()

    images, targets = _pick_vis_batch(trainer, max_samples=max_samples)

    # 这里只对可视化打开梯度，单独前向一次
    with torch.enable_grad():
        out = model.vis_compute_cem_grad(images, targets)

    S_g = out.get("S_g_CEM", None)
    if not isinstance(S_g, torch.Tensor):
        return  # 本轮没生成，直接跳过

    # 保存到 result_dir/vis/
    vis_dir = Path(trainer.args["result_dir"]) / "vis"
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(S_g.size(0)):
        heat = S_g[i].abs()  # [D] 或 [H,W]，视你 CEM 输出而定；如是向量可 reshape 或 normalize
        # 这里仅示意：做归一化并存成灰度图。你也可以复用已有的 make_overlay/colormap。
        h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        h = (h * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        save_path = vis_dir / f"epoch_{epoch:03d}_sample{i}_SgCEM.png"
        try:
            import imageio
            imageio.imwrite(save_path.as_posix(), h)
        except Exception:
            # 如果不想引入 imageio，就用 PIL 或 cv2，或改为 torch.save 二进制
            pass
