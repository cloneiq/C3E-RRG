import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from modules.causal_entanglement import CausalEntanglement

class BidirectionalEvolutionModule(nn.Module):
    """
    Dynamic Causal Bidirectional Evolution (DyCE) with population normal template θ_norm (buffer).
    Shapes (default):
      X: [B, 512], Y: [B, L, 768], F_path: [B, 512, 32, 32]
    """
    def __init__(
        self,
        K: int = 4,
        alpha: float = 0.1,
        x_dim: int = 512,
        y_dim: int = 768,
        use_learnable_l0: bool = False,
        l0: float = 0.5,
        cem_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
        chi_local_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        nhead: int = 8,
        ff_mult: int = 4,
        use_backward_net: bool = True,
        enable_early_stop: bool = False,
        stop_threshold: float = 0.02,
    ):
        super().__init__()
        self.K = int(K)
        self.alpha = float(alpha)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.lambda_gate = 1.0
        self.chi_module = None
        self._chi_grad_fn = None
        self.use_backward_net = bool(use_backward_net)
        self.enable_early_stop_default = bool(enable_early_stop)
        self.stop_threshold_default = float(stop_threshold)

        # ---- λ(0) ----
        if use_learnable_l0:
            self.l0 = nn.Parameter(torch.full((1,), float(l0)))
        else:
            self.register_buffer("l0_buf", torch.tensor(float(l0)))
            self.l0 = None

        # ---- MLPs for evolution ----
        self.mlp_micro = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, 1024), nn.GELU(),
            nn.Linear(1024, self.x_dim)
        )

        # 3 scalars -> 1 scalar
        self.mlp_lambda = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),
            nn.Linear(16, 1)
        )

        # pathology strength head and CF context head
        self.linear_p = nn.Linear(self.x_dim, 1)  # [512->1]
        self.linear_ctx_in = nn.Linear(self.x_dim * 2, self.y_dim)  # [1024->768]

        # ---- θ_norm as *buffer* (population normal template) ----
        self.register_buffer("normal_template", torch.zeros(1, self.x_dim))  # [1,512]

        # ---- CEM ----
        if cem_fn is None:
            if CausalEntanglement is None:
                raise RuntimeError("CausalEntanglement not found and cem_fn is None.")
            self._cem = CausalEntanglement(x_in_dim=self.x_dim, y_in_dim=self.y_dim)
            self._cem_fn = self._cem_forward
        else:
            self._cem = None
            self._cem_fn = cem_fn

        self._chi_local_fn = chi_local_fn or self._default_chi_local

        # ---- Minimal TransformerDecoder for 1-step expansion ----
        self.mem_proj = nn.Linear(self.y_dim, self.y_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.y_dim))
        nn.init.normal_(self.query_token, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=self.y_dim, nhead=nhead,
            dim_feedforward=self.y_dim * ff_mult,
            dropout=0.1, batch_first=True,
            activation="gelu",
        )
        self.tdec = nn.TransformerDecoder(layer, num_layers=1)

        # ---- λ smoothing & clamp ----
        self.lambda_beta = 0.2
        self.lambda_min = 0.1
        self.lambda_max = 0.9

        # y (768) -> x (512) for χ proxy
        self.y2x = nn.Linear(self.y_dim, self.x_dim)

        self.chi = None
        self.chi_grad_norm_fn = None
        self._last_cem_loss = None
        if self.use_backward_net:
            self.backward_net = nn.Sequential(
                nn.Linear(self.y_dim, 1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, self.x_dim),
            )
            self.entangle_weight_net = nn.Sequential(
                nn.Linear(self.x_dim * 3, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        else:
            self.backward_net = None
            self.entangle_weight_net = None

    # ------------------------- utils -------------------------
    def load_normal_template(self, npy_path: str):
        """
        Load θ_norm from a .npy file of shape [x_dim]. Stored as buffer (no grad).
        """
        import numpy as np
        arr = np.load(npy_path).astype("float32")
        if arr.ndim != 1 or int(arr.shape[0]) != int(self.x_dim):
            raise RuntimeError(f"normal_template.npy shape mismatch: expected [{self.x_dim}], got {arr.shape}")
        t = torch.from_numpy(arr).view(1, -1)
        self.normal_template.copy_(t.to(self.normal_template.device))
        # —— 统一尺度：用 LayerNorm 消掉量纲漂移（最小侵入）——
        self.normal_template.copy_(F.layer_norm(self.normal_template, self.normal_template.shape[-1:]))
        return True

    def set_normal_template(self, tensor: torch.Tensor):
        if tensor.dim() == 1:
            tensor = tensor.view(1, -1)
        if list(tensor.shape) != [1, self.x_dim]:
            raise RuntimeError(f"θ_norm shape must be [1,{self.x_dim}], got {list(tensor.shape)}")
        self.normal_template.copy_(tensor.to(self.normal_template.device))

    # ------------------------- CEM wrapper -------------------------
    def _cem_forward(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self._cem(X, Y)
        if not isinstance(out, dict) or ("S_g_CEM" not in out):
            raise RuntimeError("cem_fn must return dict with key 'S_g_CEM'.")
        return out

    @staticmethod
    def _masked_mean(seq: torch.Tensor) -> torch.Tensor:
        return seq.mean(dim=1)

    @staticmethod
    def _make_tgt_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def decoder_step(self, Yk: torch.Tensor, context_vec: torch.Tensor) -> torch.Tensor:
        """One-step semantic expansion.
        Args: Yk [B,L,768], context_vec [B,768]
        Returns: y_next [B,1,768]
        """
        B, L, D = Yk.size()
        q_tok = self.query_token.expand(B, 1, D)
        tgt = torch.cat([Yk, q_tok], dim=1)  # [B, L+1, D]
        tgt_mask = self._make_tgt_mask(L + 1, Yk.device)

        mem = self.mem_proj(context_vec).unsqueeze(1)  # [B,1,D]
        out = self.tdec(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
        y_next = out[:, -1:, :]
        return y_next

    # --------------------- CHI (local proxy) & ∇_Y ---------------------
    def _default_chi_local(self, F_path: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # F_path: [B,512,32,32] -> [B,512]
        f_gap = F.adaptive_avg_pool2d(F_path, (1, 1)).squeeze(-1).squeeze(-1)
        y_pool = Y.mean(dim=1)              # [B,768]
        y_proj = self.y2x(y_pool)           # [B,512]
        cos = F.cosine_similarity(F.normalize(y_proj, dim=-1), F.normalize(f_gap, dim=-1), dim=-1)
        chi = (1 - cos).clamp(min=0) ** 2   # [B]
        return chi

    def bind_chi(self, chi_module_or_fn):
        # 支持传入可调用的梯度函数，或完整的 CHI 模块
        if callable(chi_module_or_fn) and not isinstance(chi_module_or_fn, nn.Module):
            self._chi_grad_fn = chi_module_or_fn
            self.chi_module = None
        else:
            self.chi_module = chi_module_or_fn
            self._chi_grad_fn = None
            # —— 双保险：对齐 y->x 投影（若 CHI 暴露 y2x / y_proj）——
            if hasattr(self.chi_module, "y2x"):
                self.chi_module.y2x = self.y2x
            elif hasattr(self.chi_module, "core") and hasattr(self.chi_module.core, "y2x"):
                self.chi_module.core.y2x = self.y2x

    def set_chi_grad_norm_fn(self, fn):
        self._chi_grad_fn = fn

    def _chi_grad_wrt_Y(self, F_path: torch.Tensor, Y: torch.Tensor):
        if self._chi_grad_fn is not None:
            chi_val, dY_or_norm = self._chi_grad_fn(F_path, Y)
            if chi_val.dim() > 1:
                chi_val = chi_val.mean(dim=tuple(range(1, chi_val.dim())))
            return chi_val, dY_or_norm

        if self.chi_module is None:
            B = Y.size(0)
            return Y.new_zeros(B), Y.new_zeros(B)

        # 1计算图
        Yv = Y.detach().clone().requires_grad_(True)
        Fv = F_path.detach().requires_grad_(True)  
        # 2前向
        out = self.chi_module(F_path=Fv, Y_final=Yv, X_noise=getattr(self, "_x_noise", None),
                              return_heatmap=False, return_terms=False)
        if isinstance(out, dict):
            if "lambda_c" in out:
                chi = out["lambda_c"].view(-1)
            elif "H_chi" in out:
                chi = out["H_chi"].flatten(1).mean(dim=1)
            elif "chi_val" in out:
                v = out["chi_val"]
                chi = v.mean(dim=tuple(range(1, v.dim()))) if v.dim() > 1 else v
            else:
                chi = Yv.new_zeros(Yv.size(0))
        else:
            chi = out
            if chi.dim() > 1:
                chi = chi.mean(dim=tuple(range(1, chi.dim())))

        # dY = torch.autograd.grad(chi.sum(), Yv, retain_graph=False, create_graph=False)[0]
        with torch.enable_grad():                       #  显式打开梯度
            dY = torch.autograd.grad(chi.sum(), Yv, retain_graph=True, create_graph=False)[0]
        return chi.detach(), dY

    # ----------------------------- forward -----------------------------
    def forward(
            self,
            X_path_0: torch.Tensor,
            Y_0: torch.Tensor,
            F_path: torch.Tensor,
            X_noise: Optional[torch.Tensor] = None,
            K: Optional[int] = None,
            cem_override: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
            decoder_override: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            enable_early_stop: Optional[bool] = None,
            stop_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Run K steps of DyCE.
        X_path_0: [B,512], Y_0: [B,1,768], F_path: [B,512,32,32]
        """
        B = X_path_0.size(0)
        K = int(K if K is not None else self.K)
        cem_fn = cem_override or self._cem_fn
        decoder_step = decoder_override or self.decoder_step
        enable_early_stop = self.enable_early_stop_default if enable_early_stop is None else bool(enable_early_stop)
        stop_threshold = self.stop_threshold_default if stop_threshold is None else float(stop_threshold)

        self._x_noise = X_noise

        # λ init
        if self.l0 is not None:
            lamb_k = self.l0.sigmoid().expand(B)
        else:
            lamb_k = self.l0_buf.expand(B)

        Xk = X_path_0
        Yk = Y_0

        logs: List[Dict] = []
        lambda_history: List[torch.Tensor] = []
        last_X_hat: Optional[torch.Tensor] = None
        last_X_CF: Optional[torch.Tensor] = None
        last_Delta_X: Optional[torch.Tensor] = None
        early_stopped = False
        actual_K = K
        cem_loss_accum: List[torch.Tensor] = []
        for k in range(K):
            # ---- CEM: ∇X ----
            cem_out = cem_fn(Xk, Yk)              # {"S_g_CEM": [B,512], ...}
            Sg = cem_out["S_g_CEM"]
            Sg_norm = Sg.norm(p=2, dim=-1)        # [B]
            cem_loss_val = cem_out.get("cem_loss") if isinstance(cem_out, dict) else None
            if cem_loss_val is not None:
                cem_loss_accum.append(cem_loss_val)

            # ---- CHI: χ and ∇Yχ ----
            chi_val, dY = self._chi_grad_wrt_Y(F_path, Yk)   # chi_val:[B], dY:[B,L,768]
            dY_norm = dY.flatten(1).norm(p=2, dim=-1)        # [B]

            # ---- λ(k->k+1) ----
            lam_in = torch.stack([Sg_norm, dY_norm, chi_val], dim=-1)  # [B,3]
            lam_raw = torch.sigmoid(self.mlp_lambda(lam_in)).squeeze(-1)
            lamb_next = (1 - self.lambda_beta) * lamb_k + self.lambda_beta * lam_raw
            lamb_next = lamb_next.clamp(self.lambda_min, self.lambda_max)

            # ---- micro (X): update ----
            y_mean = self._masked_mean(Yk)                          # [B,768]
            F_det = self.mlp_micro(torch.cat([Xk, y_mean], -1))     # [B,512]
            lam_eff = self.lambda_gate * lamb_k

            if self.use_backward_net:
                X_hat = self.backward_net(y_mean)                   # [B,512]
                Delta_X = Xk - X_hat                                # [B,512]
                alpha_k = self.entangle_weight_net(
                    torch.cat([Xk, X_hat, Delta_X], dim=-1)
                ).squeeze(-1)                                       # [B]
                update_term = F_det + lam_eff.view(-1, 1) * Sg + alpha_k.view(-1, 1) * Delta_X
            else:
                X_hat = None
                Delta_X = None
                alpha_k = torch.zeros_like(lamb_k)
                update_term = F_det + lam_eff.view(-1, 1) * Sg

            Xk1 = Xk + self.alpha * update_term

            # ---- macro (Y): CF + one-step decode ----
            # p = torch.sigmoid(self.linear_p(Xk1))                   # [B,1]
            # theta = self.normal_template.to(Xk1).expand(B, -1)      # [B,512]
            # X_CF = (1.0 - p) * theta + p * Xk1                      # [B,512]

            # ---- pathology-strength-aware gating p(X, Y) ----
            y_pool = Yk.mean(dim=1)  # [B, Dy]
            x_from_y = self.y2x(y_pool)  # [B, Dx], 与 DyCE/CHI 共享同一 y->x 投影
            # 额外语义强度项：与 X 的逐维点积作为 logit 偏置（0.5 可按需要微调）
            p_logits = self.linear_p(Xk1) + 0.5 * (x_from_y * Xk1).sum(-1, keepdim=True)
            p = torch.sigmoid(p_logits).clamp(0.05, 0.95)  # 防止退化到 0/1

            # ---- counterfactual mixing ----
            # X_CF = (1-p)*θ_norm + p*X_k1
            theta = self.normal_template.to(Xk1).expand(Xk1.size(0), -1)  # [B, Dx]
            X_CF = (1.0 - p) * theta + p * Xk1

            ctx = self.linear_ctx_in(torch.cat([Xk1, X_CF], dim=-1))# [B,768]
            y_next = decoder_step(Yk, ctx)                          # [B,1,768]
            Yk1 = torch.cat([Yk, y_next], dim=1)

            # ---- LOG ----
            log_entry = {
                "k": k,
                "lambda_k": lamb_k.detach(),
                "lambda_k+1": lamb_next.detach(),
                "Sg_norm": Sg_norm.detach(),
                "dY_chi_norm": dY_norm.detach(),
                "chi_val": chi_val.detach(),
                "Y_len_next": torch.tensor(Yk1.size(1), device=Yk1.device),
            }
            if self.use_backward_net:
                log_entry.update({
                    "alpha_k": alpha_k.detach(),
                    "Delta_X_norm": Delta_X.norm(p=2, dim=-1).detach(),
                    "X_hat_corr": F.cosine_similarity(Xk, X_hat, dim=-1).detach(),
                })
            logs.append(log_entry)

            lambda_history.append(lamb_next.detach())
            last_X_hat = X_hat
            last_X_CF = X_CF
            last_Delta_X = Delta_X

            lamb_k, Xk, Yk = lamb_next, Xk1, Yk1
            # 在 for k in range(K) 内，更新完 Yk 之后加：
            if (k == 0) or (k == K - 1):
                if self.chi_module is not None:
                    F_path = F_path.detach().requires_grad_(True) 
                    chi_terms = self.chi_module(
                        # F_path=F_path.detach(),
                        F_path=F_path, 
                        Y_final=Yk,
                        X_noise=getattr(self, "_x_noise", None),
                        return_heatmap=True,
                        return_terms=False
                    )
                    # 仅存少量统计，避免 logs 太大
                    Hk = chi_terms.get("H_chi", None)
                    if isinstance(Hk, torch.Tensor):
                        logs.append({
                            "step": int(k),
                            "H_mean": float(Hk.mean().item()),
                            "H_std":  float(Hk.std().item())
                        })

            if enable_early_stop and len(lambda_history) >= 3:
                lam_tensor = torch.stack(lambda_history, dim=0)
                lambda_std = lam_tensor.std(dim=0).mean()
                if float(lambda_std.item()) < stop_threshold:
                    logs.append({
                        "early_stop": True,
                        "actual_K": k + 1,
                        "lambda_std": lambda_std.detach(),
                    })
                    early_stopped = True
                    actual_K = k + 1
                    break

        if not early_stopped:
            actual_K = K

        if lambda_history:
            lambda_hist_tensor = torch.stack(lambda_history, dim=0)
        else:
            lambda_hist_tensor = torch.empty(0, device=X_path_0.device)

        if cem_loss_accum:
            self._last_cem_loss = torch.stack(cem_loss_accum).mean()
        else:
            self._last_cem_loss = None

        self._last_logs = logs
        self._last_X_hat = last_X_hat
        self._last_X_CF = last_X_CF
        self._last_Delta_X = last_Delta_X
        self._last_X_path = Xk
        self._lambda_history = lambda_hist_tensor
        self._actual_K = actual_K

        return Xk, Yk, lamb_k, logs
