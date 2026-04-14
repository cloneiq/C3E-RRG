import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalEntanglement(nn.Module):
    """
    因果纠缠模块（CEM）
    -----------------------------------------
    输入:
      X_path : [B, 512]   —— 视觉病理特征
      Y      : [B, L, 768] —— 文本特征序列
    输出:
      {
        "cem_value": [B],       # 因果纠缠值（监控）
        "S_g_CEM":   [B, 512],  # 对 X 的梯度场（指导信号）
        "cem_loss":  标量(仅训练态)
      }
    -----------------------------------------
    原理：
      CEM = 有界化核范数 + λ * 熵正则
      训练时执行干预 (shuffle Y)，cem_loss = -(CEM - CEM_do)
    """

    def __init__(self, x_in_dim=512, y_in_dim=768,
                 proj_dim=256, lambda_ent=0.1, eps=1e-8,
                 act="gelu", nuc_scale=5.0,
                 nuclear_norm_mode: str = "bounded_fro",
                 schatten_p: float = 1.0,
                 svd_lowrank_q: int = 32,
                 use_sv_entropy: bool = True,
                 sv_entropy_weight: float = 0.7,
                 adaptive_lambda_ent: bool = True):
        super().__init__()
        self.x_in_dim   = int(x_in_dim)
        self.y_in_dim   = int(y_in_dim)
        self.proj_dim   = int(proj_dim)
        self.base_lambda_ent = float(lambda_ent)
        self.eps        = float(eps)
        self.nuc_scale  = float(nuc_scale)
        self.nuclear_norm_mode = (nuclear_norm_mode or "bounded_fro").lower()
        self.schatten_p = float(max(schatten_p, 1e-6))
        self.svd_lowrank_q = int(max(svd_lowrank_q, 1))
        self.use_sv_entropy = bool(use_sv_entropy)
        self.sv_entropy_weight = float(min(max(sv_entropy_weight, 0.0), 1.0))
        self.adaptive_lambda_ent = bool(adaptive_lambda_ent)
        self.register_buffer("current_epoch", torch.zeros(1, dtype=torch.long))

        # 投影层
        self.linear_phi = nn.Linear(self.x_in_dim, self.proj_dim)
        # 条件化投影层：当提供 Z（[B,512]）时使用
        self.linear_phi_cond = nn.Linear(self.x_in_dim + 512, self.proj_dim)
        self.linear_psi = nn.Linear(self.y_in_dim, self.proj_dim)

        # 激活函数
        act = (act or "gelu").lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """由 Trainer 在每个 epoch 开始时调用，控制自适应 λ_ent。"""
        epoch = int(epoch)
        self.current_epoch.fill_(max(epoch, 0))

    def _get_lambda_ent(self) -> float:
        if not self.adaptive_lambda_ent:
            return self.base_lambda_ent
        epoch = int(self.current_epoch.item())
        if epoch < 5:
            return 0.05
        if epoch < 15:
            return 0.1
        return 0.15

    def _sanitize_outer(self, outer: torch.Tensor) -> torch.Tensor:
        """
        清洗 outer 中的 NaN / Inf，避免 SVD 直接炸掉。
        """
        if torch.isfinite(outer).all():
            return outer

        outer = outer.clone()
        bad_mask = ~torch.isfinite(outer)
        if bad_mask.any():
            outer[bad_mask] = 0.0
            # 只提示一次，避免刷屏
            if not hasattr(self, "_warned_nan") or not self._warned_nan:
                print("[CEM][WARN] outer had NaN/Inf; sanitized before SVD.", flush=True)
                self._warned_nan = True
        return outer

    def _nuclear_norm(self, outer: torch.Tensor) -> torch.Tensor:
        """
        outer: [B, proj_dim, proj_dim]
        返回: [B]
        """
        # 先清洗 NaN / Inf
        outer = self._sanitize_outer(outer)
        # 防止数值过大
        outer = outer.clamp(min=-1e4, max=1e4)

        if self.nuclear_norm_mode == "bounded_fro":
            nuc_unbounded = outer.norm(p='fro', dim=(1, 2))
            return nuc_unbounded / (1.0 + nuc_unbounded / self.nuc_scale)

        # 下面几种模式要用 SVD
        try:
            sigma = torch.linalg.svdvals(outer)  # [B, proj_dim]
        except RuntimeError:
            try:
                # 再试一次常规 SVD
                sigma = torch.linalg.svd(outer, full_matrices=False).S
            except RuntimeError:
                # 最后兜底：搬到 CPU 上算 SVD，避免 GPU cusolver BUG
                outer_cpu = outer.detach().to(dtype=torch.float64, device="cpu")
                sigma_cpu = torch.linalg.svdvals(outer_cpu)
                sigma = sigma_cpu.to(outer.device, dtype=outer.dtype)

        if self.nuclear_norm_mode == "schatten_p":
            p = self.schatten_p
            if math.isclose(p, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                return sigma.sum(dim=-1)
            sigma_p = sigma.pow(p)
            return sigma_p.sum(dim=-1).pow(1.0 / p)

        if self.nuclear_norm_mode == "svd_lowrank":
            q = min(self.svd_lowrank_q, sigma.size(-1))
            sigma_topk = sigma[..., :q]
            return sigma_topk.sum(dim=-1)

        raise ValueError(f"Unknown nuclear_norm_mode: {self.nuclear_norm_mode}")

    def _singular_value_entropy(self, outer: torch.Tensor) -> torch.Tensor:
        """
        奇异值熵: H = -Σ(σ_i_norm * log σ_i_norm)
        outer: [B, proj_dim, proj_dim]
        返回: [B]
        """
        # 清洗 NaN / Inf，避免 SVD 爆炸
        outer = self._sanitize_outer(outer)
        outer = outer.clamp(min=-1e4, max=1e4)

        try:
            sigma = torch.linalg.svdvals(outer)
        except RuntimeError:
            try:
                sigma = torch.linalg.svd(outer, full_matrices=False).S
            except RuntimeError:
                outer_cpu = outer.detach().to(dtype=torch.float64, device="cpu")
                sigma_cpu = torch.linalg.svdvals(outer_cpu)
                sigma = sigma_cpu.to(outer.device, dtype=outer.dtype)

        sigma_sum = sigma.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        sigma_norm = sigma / sigma_sum
        entropy = -(sigma_norm * (sigma_norm + self.eps).log()).sum(dim=-1)
        return entropy

    # ------------------------------------------------------------
    # 核心计算：CEM = bounded nuclear norm + entropy regularizer
    # ------------------------------------------------------------
    def _cem_core(self, X_path: torch.Tensor, Y: torch.Tensor, Z: Optional[torch.Tensor] = None):
        """
        输入:
          X_path: [B, D_x]
          Y: [B, L, D_y]
          Z: [B, 512] 或 None
        输出:
          cem_value: [B]
          phi, psi: [B, D_proj]
        """
        B = X_path.size(0)

        # 线性 + 激活（支持条件化）
        if Z is not None:
            phi_in = torch.cat([X_path, Z], dim=-1)          # [B, x_dim+512]
            phi_raw = self.act(self.linear_phi_cond(phi_in)) # [B, D_proj]
        else:
            phi_raw = self.act(self.linear_phi(X_path))      # [B, D_proj]
        psi_raw = self.act(self.linear_psi(Y.mean(dim=1)))   # [B, D_proj]

        # 方向归一化
        phi = F.normalize(phi_raw, p=2, dim=-1)
        psi = F.normalize(psi_raw, p=2, dim=-1)

        # ==== 计算外积矩阵 ====
        outer = (phi.unsqueeze(2) @ psi.unsqueeze(1))        # [B, D, D]

        # ==== 核范数（可选近似）====
        nuclear_norm = self._nuclear_norm(outer)             # [B]

        # ==== 熵正则：跨样本注意力 ====
        # 使用跨模态相似度矩阵而非逐维乘积
        attn_score = torch.matmul(phi, psi.T) / math.sqrt(self.proj_dim)  # [B, B]
        P = F.softmax(attn_score, dim=-1)
        attn_entropy = -(P * (P + self.eps).log()).sum(dim=-1)  # [B]

        if self.use_sv_entropy:
            sv_entropy = self._singular_value_entropy(outer)
            weight = self.sv_entropy_weight
            total_entropy = weight * sv_entropy + (1.0 - weight) * attn_entropy
        else:
            total_entropy = attn_entropy

        lambda_ent = self._get_lambda_ent()

        # ==== 综合 ====
        cem_value = nuclear_norm + lambda_ent * total_entropy
        return cem_value, phi, psi

    # ------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------
    def forward(self, X_path: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor = None):
        # 检查输入维度
        assert X_path.dim() == 2 and X_path.size(-1) == self.x_in_dim, \
            f"X_path shape error: expected [B,{self.x_in_dim}] got {tuple(X_path.shape)}"
        assert Y.dim() == 3 and Y.size(-1) == self.y_in_dim, \
            f"Y shape error: expected [B,L,{self.y_in_dim}] got {tuple(Y.shape)}"

        # 确保 X 可求梯度
        if not X_path.requires_grad:
            X_path.requires_grad_(True)

        # 正常计算 CEM
        cem_xy, phi, psi = self._cem_core(X_path, Y, Z=Z)  # [B], [B,D], [B,D]

        # 对 X 求梯度场（∂CEM/∂X）
        S_g = torch.autograd.grad(
            cem_xy.sum(), X_path,
            retain_graph=self.training, create_graph=False
        )[0]

        # 输出结果字典
        out = {
            "cem_value": cem_xy.detach(),  # 监控用
            "S_g_CEM":  S_g,               # 指导信号（保留梯度）
        }

        # ============================================================
        # 训练态下执行干预实验 (do-shuffle)
        # ============================================================
        if self.training:
            B = Y.size(0)
            with torch.no_grad():
                perm = torch.randperm(B, device=Y.device)
                Y_shuf = Y.index_select(0, perm)

            # 注意：仅 Y 打乱；Z 若提供则不打乱（保持配对）
            cem_do, _, _ = self._cem_core(X_path, Y_shuf, Z=Z)
            cem_loss = -(cem_xy - cem_do).mean()  # 最大化差分
            out["cem_loss"] = cem_loss
        else:
            out["cem_loss"] = None

        return out
