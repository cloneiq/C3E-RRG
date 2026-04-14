import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class DepthwiseSeparableConv(nn.Module):
    """轻量卷积块：深度可分离卷积"""
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

    def forward(self, x):  # [B,C,H,W] -> [B,out,H,W]
        x = self.dw(x)
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class CausalHollowIndexModule(nn.Module):
    """
    稳健版 CHI：
      - term1 使用 SmoothGrad 思想：对 Y 做 T 次高斯微扰，平均解析梯度范数的倒数 -> 降噪
      - term2 输出后做 3x3 低通平滑，初始化为偏低（-bias）防止随机纹理
      - H_chi 输出前做轻量 3x3 平滑（不改变均值，几乎不影响 lambda_c）
      - chi_avg 支持 mean / pXX（如 p95）聚合（可缓解孤立亮点）
      - 可选 M_anat 掩码（只影响 H_chi，不改变内部梯度）
    """
    def __init__(
        self,
        f_dim: int = 512,
        y_dim: int = 768,
        grid_hw: Tuple[int, int] = (32, 32),
        eps: float = 1e-6,
        frag_mid_ch: int = 64,
        sg_T: int = 4,               # SmoothGrad 次数
        sg_sigma: float = 0.1,       # 扰动强度
        smooth_kernel: int = 3,      # H_chi 平滑核
        frag_smooth_kernel: int = 3, # term2 平滑核
        chi_pool: str = "mean",      # "mean" | "p95"
    ):
        super().__init__()
        self.f_dim, self.y_dim = f_dim, y_dim
        self.Hf, self.Wf = grid_hw
        self.N = self.Hf * self.Wf
        self.eps = eps
        self.sg_T = int(sg_T)
        self.sg_sigma = float(sg_sigma)
        self.smooth_k = int(smooth_kernel)
        self.frag_smooth_k = int(frag_smooth_kernel)
        self.chi_pool = chi_pool

        # --- 映射层 --- #
        self.y_proj = nn.Linear(y_dim, f_dim, bias=False)
        self.y2x = self.y_proj
        self.lin_y2x = self.y_proj

        # --- 脆弱性网络（用于特征空间） --- #
        self.fragility_net = nn.Sequential(
            nn.Conv2d(f_dim, f_dim, 1, bias=False),
            nn.BatchNorm2d(f_dim),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(f_dim, frag_mid_ch, 1),
            nn.Sigmoid()
        )
        # 显式初始化最后层偏置为 -2.0（sigmoid(-2)=0.12）
        if hasattr(self.fragility_net[-2], "pw2"):
            with torch.no_grad():
                self.fragility_net[-2].pw2.bias.fill_(-2.0)
        
        # --- 脆弱性网络（用于语义空间，从y_pool计算） --- #
        self.fragility_mlp = nn.Sequential(
            nn.Linear(y_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Sigmoid()
        )

        self.noise_mlp = nn.Sequential(
            nn.Linear(f_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Sigmoid()
        )

        self.conf_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Lambda网络: [CHI, 脆弱性, 噪声] → 置信度
        self.lambda_net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    # ---------------------- 工具函数 ---------------------- #
    @staticmethod
    def _l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
        return torch.clamp(x.norm(p=2, dim=dim, keepdim=True), min=eps)

    def _consistency(self, Y: torch.Tensor) -> torch.Tensor:
        y_bar = Y.mean(dim=1)
        Y_n = Y / self._l2_norm(Y, -1, self.eps)
        yb_n = y_bar / self._l2_norm(y_bar, -1, self.eps)
        cos = (Y_n * yb_n.unsqueeze(1)).sum(-1)
        return ((cos + 1.) * 0.5).mean(dim=1)

    def _noise_entropy(self, X_noise: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(X_noise, dim=-1)
        ent = -(p * torch.clamp(p, min=self.eps).log()).sum(-1)
        ent = ent / torch.log(torch.tensor(X_noise.size(-1), device=X_noise.device, dtype=X_noise.dtype))
        return torch.clamp(ent, 0.0, 1.0)

    # ---------------------- term1: gradient inverse norm ---------------------- #
    def _term1_grad_inv_norm_once(self, F_path: torch.Tensor, Y_pooled: torch.Tensor) -> torch.Tensor:
        """ term1 = 1 / (1 + ||∂cos(f̂,ŷ)/∂f||)，平滑稳定版 """
        B, C, H, W = F_path.shape
        F_flat = F_path.flatten(2).transpose(1, 2)  # [B,N,C]
        f_norm = self._l2_norm(F_flat, -1, self.eps)
        f_hat = F_flat / f_norm

        y_512 = self.y_proj(Y_pooled)
        y_hat = y_512 / self._l2_norm(y_512, -1, self.eps)

        s = (f_hat * y_hat.unsqueeze(1)).sum(-1, keepdim=True)  # [B,N,1]
        grad_vec = y_hat.unsqueeze(1) - s * f_hat
        grad_norm = self._l2_norm(grad_vec, -1, self.eps) / torch.clamp(f_norm, min=self.eps)
        term1 = 1.0 / (1.0 + grad_norm)  # ✅ 稳定替代方案
        return term1.squeeze(-1).reshape(B, H, W)

    def _term1_smoothgrad(self, F_path: torch.Tensor, Y_pooled: torch.Tensor) -> torch.Tensor:
        """ 对 Y 做 T 次微扰并平均 """
        if self.sg_T <= 1:
            return self._term1_grad_inv_norm_once(F_path, Y_pooled)
        B, D = Y_pooled.shape
        out = 0.0
        for _ in range(self.sg_T):
            noise = torch.randn_like(Y_pooled) * (self.sg_sigma * (Y_pooled.std(dim=-1, keepdim=True) + 1e-3))
            out = out + self._term1_grad_inv_norm_once(F_path, Y_pooled + noise)
        return out / float(self.sg_T)

    # ---------------------- 前向传播 ---------------------- #
    def forward(self,
                F_path: torch.Tensor,
                Y_final: torch.Tensor,
                X_noise: Optional[torch.Tensor] = None,
                return_heatmap: bool = True,
                return_terms: bool = False) -> Dict[str, torch.Tensor]:
        """
        通过基于梯度的因果干预计算因果空洞指数(CHI)
        
        核心原理:
        - 梯度 ∂(align_loss)/∂F[i,j] 反映空间位置(i,j)对语义的因果贡献
        - 高梯度 = 强因果作用(如病灶) → CHI深色
        - 低梯度 = 弱/无因果作用(如背景) → CHI浅色
        理论依据:
        - 梯度是目标函数关于输入的局部敏感性
        - 在因果推理中,敏感性≈因果影响强度
        - 等价于边际干预效应: ∂P(Y|X)/∂X[i,j]
        """
        assert F_path.dim() == 4, f"F_path shape expected [B,C,H,W], got {tuple(F_path.shape)}"
        B, C, H, W = F_path.shape
        device = F_path.device
        
        # ========== 步骤1: 语义投影 ==========
        y_pool = Y_final.mean(dim=1)  # [B, D_y] 序列平均
        y_proj = self.y2x(y_pool) if hasattr(self, "y2x") else self.lin_y2x(y_pool)  # [B, D_x=512]
        
        # ========== 步骤2: 因果影响计算(核心) ==========
        # 在 enable_grad 上下文中克隆并重新计算，确保有计算图
        with torch.enable_grad():
            # 克隆 F_path 以确保在新的计算图中
            F_var = F_path.clone().requires_grad_(True)
            # 语义-特征对齐损失(可微目标)
            f_gap = F.adaptive_avg_pool2d(F_var, (1, 1)).squeeze(-1).squeeze(-1)  # [B, D]
            # y_proj 可以保留梯度，但我们只关心 F_var 的梯度
            align_loss = F.mse_loss(y_proj, f_gap, reduction='sum')
            
            # 计算梯度: ∂(align_loss)/∂F[i,j] for all (i,j)
            grad_F = torch.autograd.grad(
                outputs=align_loss,
                inputs=F_var,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]  # [B, D, H, W]

        # 因果图 = 梯度L2范数(通道维度)
        # 物理意义: 该位置所有通道的因果贡献总和
        H_chi = grad_F.norm(p=2, dim=1)  # [B, H, W]
        
        # 归一化到[0,1](每个样本独立归一化)
        H_min = H_chi.amin(dim=(1, 2), keepdim=True)
        H_max = H_chi.amax(dim=(1, 2), keepdim=True)
        H_chi = (H_chi - H_min) / (H_max - H_min + self.eps)

        # ========== 步骤3: 语义引导平滑(可选) ==========
        if self.smooth_k and self.smooth_k > 1:
            pad = self.smooth_k // 2
            H_chi = F.avg_pool2d(
                H_chi.unsqueeze(1), kernel_size=self.smooth_k, stride=1, padding=pad
            ).squeeze(1)
        
        # ========== 步骤4: 脆弱性计算 ==========
        frag_scalar = self.fragility_mlp(y_pool)  # [B,1] MLP
        frag_mean = frag_scalar.squeeze(-1)       # [B]
        
        # ========== 步骤5: 置信度门控 lambda_c ==========
        H_mean = H_chi.flatten(1).mean(dim=1)  # [B]
        
        # 噪声感知
        if X_noise is not None and X_noise.dim() == 2 and X_noise.size(-1) == C:
            eta = self.noise_mlp(X_noise).squeeze(-1)  # [B]
        else:
            f_gap_det = F_path.detach().mean(dim=(2, 3))  # [B,512]
            eta = self.noise_mlp(f_gap_det).squeeze(-1)  # [B]
        
        # 池化策略
        if self.chi_pool == "p95":
            H_pooled = torch.quantile(H_chi.flatten(1), 0.95, dim=1)
        elif self.chi_pool == "max":
            H_pooled = H_chi.flatten(1).max(dim=1).values
        else:  # "mean"
            H_pooled = H_mean
        
        # Lambda网络: [CHI, 脆弱性, 噪声] → 置信度
        lam_in = torch.stack([H_pooled, frag_mean, eta], dim=-1)  # [B,3]
        lambda_c = torch.sigmoid(self.lambda_net(lam_in)).squeeze(-1)  # [B]
        
        # 反转: 高CHI → 低置信度(空洞多 → 不可靠)
        lambda_c = 1.0 - lambda_c
        lambda_c = lambda_c.clamp(0.05, 0.95)
        
        # ========== 步骤6: 上采样到224x224 ==========
        H_chi_224 = F.interpolate(
            H_chi.unsqueeze(1), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # [B, 224, 224]
        
        # ========== 输出 ==========
        out = {
            "H_chi": H_chi_224,        # [B,224,224] 因果热力图(上采样)
            "lambda_c": lambda_c,     # [B] 置信度
        }
        
        if return_terms:
            out["terms"] = {
                "H_mean": H_mean,
                "H_pooled": H_pooled,
                "frag": frag_mean,
                "eta": eta,
                "grad_norm_mean": grad_F.flatten(1).norm(dim=1).mean(),
                "H_chi_32": H_chi,  # 保留原始32x32版本
            }
        
        return out

