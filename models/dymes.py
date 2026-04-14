import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .baseline import Baseline
from modules.modules4vlp import get_ht_mask, get_hv_mask, get_cross_mask
from modules.modules4transformer import LayerNorm, DecoderLayer
from modules.feature_disentanglement import CausalEncoder, LGFM
from modules.confounder_modeling import ConfounderEncoder
from modules.causal_entanglement import CausalEntanglement
from modules.bidirectional_evolution import BidirectionalEvolutionModule
from modules.causal_hollow_index import CausalHollowIndexModule

# ====== 单解码器：在进入层堆前注入后门门控（Z）与融合的 cross-memory（hv⊕ctx_y） ======
class CausalDecoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)
        self.ht = None  # for visualization
        # 轻量的“语言去偏”门控（句级门控）
        self.z_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def _backdoor_gate(self, ht, z):
        # ht: [B,L,D], z: [B,D] or None
        if z is None:
            return ht, 0.0
        h_pool = ht.mean(dim=1)                                # [B,D]
        g = self.gate_mlp(torch.cat([h_pool, z], dim=-1))      # [B,1]∈(0,1)
        ht = ht + g.unsqueeze(1) * self.z_proj(z).unsqueeze(1) # 句级注入（抑制语言先验）
        return ht, g.mean().detach()

    def forward(self, ht, memory, self_mask=None, cross_mask=None, z=None):
        # 语言侧去偏（Backdoor 门控）
        ht, gate_val = self._backdoor_gate(ht, z)
        self.ht = ht

        # 常规解码层堆（与 Baseline 同构）
        for layer in self.layers:
            ht = layer(ht, memory, self_mask, cross_mask)

        out = self.norm(ht)
        return out, float(gate_val)

# ====== DYMES 主体：继承 Baseline，重写 decoder 为 CausalDecoder，并在 Enc/Dec 之间接入因果支路 ======
class DYMES(Baseline):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

        # --- 用自己的 CausalDecoder（单解码器）覆盖 Baseline 的 decoder ---
        self.decoder = CausalDecoder(
            embed_dim=self.embed_dim,
            num_layer=self.de_num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout
        )
        # --- 因果侧路组件（兼容三处：顶层 / model 段 / dymes 段） ---
        x_dim = int(args.get("x_dim", 512))
        z_dim = int(args.get("z_dim", 512))
        self.z_dim = z_dim
        self.z_gate_proj = nn.Linear(self.z_dim, self.embed_dim)
        self.ctx_tokens = int(args.get("ctx_tokens", 4))
        self.K = int(args.get("K", 4))
        self.use_backward_net = bool(args.get("use_backward_net", True))
        self.enable = bool(args.get("enable", True))
        # 特征编码的 CausalEncoder 替换 Baseline 的 Encoder
        self.encoder = CausalEncoder(
            embed_dim=self.embed_dim,
            num_layer=self.en_num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mode=self.args.get("v_causal", "n"),  # 用 v_causal 控制 :contentReference[oaicite:8]{index=8}
        )
        # 用 LGFM 来 fusion fl / fg 得到 Mv
        self.lgfm = LGFM(self.embed_dim)
        # 混淆因子：路径缺失/文件不存在 => 自动禁用（不报错）
        pmi_path  = args.get("pmi_matrix_path", "")
        corr_path = args.get("disease_corr_path", "")
        self.conf = None
        if pmi_path and corr_path and os.path.exists(pmi_path) and os.path.exists(corr_path):
            self.conf = ConfounderEncoder(
                pmi_matrix_path=pmi_path,
                disease_corr_path=corr_path,
                z_dim=z_dim
            )
        else:
            print("[DYMES][WARN] confounder disabled (missing pmi_matrix_path/disease_corr_path).")

        # ====== dymes.py 片段：__init__ 中 CEM 的构造（解耦内部/外部） ======
        self.cem = CausalEntanglement(
            x_in_dim=x_dim, y_in_dim=self.embed_dim,
            proj_dim=int(args.get("cem_proj_dim", 256)),
            nuclear_norm_mode=str(args.get("cem_nuclear_norm_mode", "bounded_fro")),
            schatten_p=float(args.get("cem_schatten_p", 1.0)),
            svd_lowrank_q=int(args.get("cem_svd_lowrank_q", 32)),
            use_sv_entropy=bool(args.get("cem_use_sv_entropy", False)),
            sv_entropy_weight=float(args.get("cem_sv_entropy_weight", 0.7)),
            adaptive_lambda_ent=bool(args.get("cem_adaptive_lambda_ent", False)),
            lambda_ent=float(args.get("cem_lambda_ent", args.get("lambda_cem", 0.05)))  # ← 新增 cem_lambda_ent，向后兼容
        )
        
        # DyCE
        self.dyce = BidirectionalEvolutionModule(
            K=self.K, alpha=float(args.get("alpha_micro", 0.05)),
            x_dim=x_dim, y_dim=self.embed_dim, use_backward_net=self.use_backward_net
        )
        nt_path = args.get("normal_template_path", "")
        if nt_path:
            try:
                self.dyce.load_normal_template(nt_path)
            except Exception as e:
                print(f"[DYMES][WARN] load normal_template failed: {e}")

        # CHI
        self.use_chi = bool(args.get("enable_chi", False))
        grid_hw = int(args.get("grid_hw", 32))
        self.chi = CausalHollowIndexModule(f_dim=x_dim, y_dim=self.embed_dim,
                                           grid_hw=(grid_hw, grid_hw)) if self.use_chi else None
        if self.chi is not None:
            self.dyce.bind_chi(self.chi)
        
        # === 语义 seed 与 ctx 投影（缺失就会 AttributeError）===
        self.seed_mlp = nn.Sequential(
            nn.Linear(x_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        # 损失权重
        self.lambda_cem      = float(args.get("lambda_cem", 0.0))
        self.lambda_back     = float(args.get("lambda_back", 0.0))
        self.lambda_cf       = float(args.get("lambda_cf", 0.0))
        self.lambda_fd_ortho = float(args.get("lambda_fd_ortho", 0.0))

        # 训练步计数（给 trainer 存 CHI 图用）
        self._global_step = 0

        # 需要的数据集分支
        if args["dataset_name"] == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args["dataset_name"] == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr

    # -------- IU-Xray：双视角主干 + 因果侧路 --------
    def forward_iu_xray(self, images, targets=None, mode='train'):
        """
        images: [B, 2, 3, 224, 224]
        """
        B, V, C, H, W = images.shape
        # Baseline 视觉主干分支
        hv = self.vis_embed(images.reshape(B*V, 3, H, W))        # [B*2,N,D]
        outputs = self._forward_core(hv, targets, mode, B, images)
        return outputs

    # -------- MIMIC-CXR：单视角主干 + 因果侧路 --------
    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        """
        images: [B, 3, 224, 224]
        """
        B, C, H, W = images.shape
        hv = self.vis_embed(images)                              # [B,N,D]
        outputs = self._forward_core(hv, targets, mode, B, images)
        return outputs

    def _get_Z(self, token_ids: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor:
        """
        统一的 Z 获取：训练/推理都用 conf 估计（推理不打乱）
        Args:
            token_ids: [B, T] 训练时可用 GT，采样时用“当前已生成前缀”
            mode: 'train' | 'sample' | 'val'
        """
        if self.conf is None or token_ids is None:
            # 推理或无文本时，返回全零 Z（保尺寸）
            B = token_ids.size(0) if isinstance(token_ids, torch.Tensor) else 1
            return torch.zeros(B, self.z_dim, device=device)
        return self.conf(token_ids, training=(mode == "train"))

    # ---------------- 核心 ----------------
    def _forward_core(self, hv, targets, mode, B, raw_images=None):
        # ===== 1. 视觉编码：hv -> CausalEncoder -> mediator(fl, fg) =====
        # reshape patch tokens + 加 CLS
        hv = hv.reshape([B, -1, self.embed_dim])                # [B, Tv-1, D]
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)                 # [B, Tv, D]
        # CausalEncoder
        hv_mask = get_hv_mask(hv)
        proj_flag = bool(self.args.get("causal_proj", False))
        hv, mediator = self.encoder(hv, hv_mask, self.vis_embed.pos_embed, proj=proj_flag)
        fl = mediator.get("local")   # [B, L_l, D] 局部采样特征
        fg = mediator.get("global")  # [B, L_g, D] 全局采样特征
        # ===== 2. 把 fl/fg/Mv 映射成 X_path / F_path / X_noise =====
        # 2.1 Mv = LGFM(fl, fg)，再对 token 做平均 → X_path 向量
        mv = self.lgfm(fl, fg)                      # [B, L_l, D]
        X_path = mv.mean(dim=1)                     # [B, D] → Xp / Xpath
        # 2.2 X_noise 用全局采样 fg 的平均
        X_noise = fg.mean(dim=1)                    # [B, D] → Xn / Xnoise
        # 2.3 F_path 用局部采样 fl，reshape 成伪 2D，并插值到 32×32
        B_l, L_l, D_l = fl.shape                    # L_l 一般是 k*heads，例如 6*8=48 :contentReference[oaicite:14]{index=14}
        # 尽量拆成接近正方形的 H×W，例如 48 -> 6×8
        H = int(L_l ** 0.5)
        while H > 1 and (L_l % H) != 0:
            H -= 1
        W = max(1, L_l // H)
        F_path = fl.transpose(1, 2).reshape(B_l, D_l, H, W)      # [B, D, H, W]
        # 为了和之前 FD 输出兼容，统一插值到 [B,512,32,32]，方便 DyCE / CHI 使用 :contentReference[oaicite:15]{index=15}
        F_path = F.interpolate(F_path, size=(32, 32),
                               mode="bilinear", align_corners=False)

        aux = {
            "fd_x_path": X_path,    # 保持键名不变，loss 里会拿来做正交约束 :contentReference[oaicite:16]{index=16}
            "fd_x_noise": X_noise,
        }
        chi_map = None
        z_for_gate = None
        ctx_y = None

        # ===== 3. 因果侧路：Z → CEM → DyCE（训练态 + enable=True 才开启） =====
        if mode == "train" and self.enable:
            # 3.1 显式混淆因子 Z（后门打乱由 ConfounderEncoder 控制）:contentReference[oaicite:17]{index=17}
            Z = self._get_Z(targets, mode, hv.device)           # [B, z_dim]
            # 3.2 从 X_path 映射到文本空间，作为起始语义 seed
            X_path_cem = X_path.detach().requires_grad_(True)   # 给 CEM 求 ∂CEM/∂X 用
            Y_seed = self.seed_mlp(X_path_cem).unsqueeze(1)     # [B,1,D]
            z_for_gate = self.z_gate_proj(Z.detach())           # 解码器后门门控

            # 3.3 CEM：计算 cem_value + S_g_CEM + cem_loss :contentReference[oaicite:18]{index=18}
            cem_out = self.cem(X_path_cem, Y_seed, Z=Z)
            S_g_CEM = cem_out["S_g_CEM"]                        # [B, D]
            cem_loss = cem_out.get("cem_loss", None)
            aux["cem_value"] = cem_out.get("cem_value", None)

            # 3.4 DyCE：K 步动态双向演化（用 F_path, X_noise, S_g_CEM）:contentReference[oaicite:19]{index=19}
            X_fin, Y_fin, lambda_fin, logs = self.dyce(
                X_path_0=X_path_cem,
                Y_0=Y_seed,
                F_path=F_path,
                X_noise=X_noise,
                cem_override=(lambda X, Y: {"S_g_CEM": S_g_CEM}),
            )

            # 取最后 C 个 token 作为语义 ctx
            C = min(self.ctx_tokens, Y_fin.size(1))
            ctx_y = self.ctx_proj(Y_fin[:, -C:, :])             # [B, C, D]

            # 3.5 CHI：绑定了 CHI，就直接用 dyce 内部的 H_chi，否则 fallback 用 F_path 能量 :contentReference[oaicite:20]{index=20}
            if self.use_chi and hasattr(self.dyce, "_last_chi_map"):
                chi = self.dyce._last_chi_map
                if isinstance(chi, torch.Tensor):
                    chi_map = chi.detach().cpu()
            if chi_map is None:
                try:
                    chi_fallback = F_path.pow(2).sum(dim=1)     # [B,32,32]
                    chi_map = chi_fallback.detach().cpu()
                except Exception:
                    chi_map = None
        else:
            # 验证 / 推理
            Z = torch.zeros(B, self.z_dim, device=hv.device)
            Y_seed = self.seed_mlp(X_path).unsqueeze(1)             # [B,1,D]
            ctx_y = self.ctx_proj(Y_seed).repeat(1, self.ctx_tokens, 1)
            cem_loss = None

        # ===== 4. hv 与 ctx_y 融合，进入单解码器 =====
        if mode == "train" and ctx_y is not None:
            hv_aug = torch.cat([hv, ctx_y], dim=1)  # 在视觉 memory 后拼上因果演化出来的语义 token
        else:
            hv_aug = hv

        if mode == "train":
            self._global_step += 1
            cross_mask = get_cross_mask(hv_aug, targets)
            ht_mask, targets = get_ht_mask(targets)
            ht = self.text_embed(targets)                            # [B,L,D]
            out, z_gate_val = self.decoder(
                ht, hv_aug,
                self_mask=ht_mask,
                cross_mask=cross_mask,
                z=z_for_gate
            )
            log_probs = F.log_softmax(self.logit(out), dim=-1)

            # 记录可视化用信息（CHI + DyCE 中间量）
            if isinstance(chi_map, torch.Tensor):
                self._last_H_chi = chi_map
                try:
                    self._last_chi_stats = (
                        float(chi_map.min()),
                        float(chi_map.max()),
                        float(chi_map.std()),
                    )
                except Exception:
                    self._last_chi_stats = None

            self._X_hat        = getattr(self.dyce, "_last_X_hat", None)
            self._X_path_final = getattr(self.dyce, "_last_X_path", None)
            self._X_CF         = getattr(self.dyce, "_last_X_CF", None)

            meta = {
                "cem_loss": cem_loss,
                "fd_x_path": aux.get("fd_x_path"),
                "fd_x_noise": aux.get("fd_x_noise"),
                "chi_map": chi_map,
                "global_step": self._global_step,
                "ht": self.decoder.ht.detach(),
                "memory": hv_aug.detach(),
            }
            return [log_probs, {}, meta]

        elif mode == "sample":
            # ===== 5. 采样
            def _sample_forward(hv_tokens, ht_ids, v_mask, t_mask):
                ht = self.text_embed(ht_ids)
                z_eval = None
                if self.conf is not None:
                    with torch.no_grad():
                        Z_s = self._get_Z(ht_ids, mode="sample", device=ht_ids.device)
                    z_eval = self.z_gate_proj(Z_s)

                with torch.no_grad():
                    X_path_approx = hv_tokens.mean(dim=1)        # 用视觉 memory 均值近似 X_path
                    Y_seed_s = self.seed_mlp(X_path_approx)
                    ctx_y_s = self.ctx_proj(Y_seed_s)
                ctx_rep = ctx_y_s.unsqueeze(1).repeat(1, self.ctx_tokens, 1)
                memory = torch.cat([hv_tokens, ctx_rep], dim=1)

                if v_mask is not None:
                    if v_mask.dim() == 2:
                        Bm = v_mask.size(0)
                        vm_ctx = torch.ones(Bm, self.ctx_tokens, device=v_mask.device, dtype=v_mask.dtype)
                        cross_mask = torch.cat([v_mask, vm_ctx], dim=1)
                    else:
                        ones_shape = list(v_mask.size())
                        ones_shape[-1] = self.ctx_tokens
                        vm_ctx = torch.ones(*ones_shape, device=v_mask.device, dtype=v_mask.dtype)
                        cross_mask = torch.cat([v_mask, vm_ctx], dim=-1)
                else:
                    cross_mask = None

                out, _ = self.decoder(ht, memory,
                                      self_mask=t_mask,
                                      cross_mask=cross_mask,
                                      z=z_eval)
                return out

            self.beam_search.load_model(_sample_forward, self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
            try:
                self._last_report_text = self.tokenizer.decode_batch(
                    outputs.detach().cpu().numpy()
                )[0]
            except Exception:
                self._last_report_text = ""
            return outputs
        else:
            raise ValueError("mode must be 'train' or 'sample'")

    def vis_compute_cem_grad(self, images, targets=None):
        """
        仅用于可视化：用当前因果编码器视觉分支 + CEM，算 ∂CEM/∂X_path。
        """
        self.eval()
        device = images.device
        B = images.size(0)

        # 1) 视觉 embed（和 forward 一样）
        if images.dim() == 5:  # [B,2,3,224,224]
            B, V, C, H, W = images.shape
            im = images.view(B * V, C, H, W)
            hv = self.vis_embed(im)  # [B*V, N, D]
        else:  # [B,3,224,224]
            hv = self.vis_embed(images)  # [B, N, D]
            V = 1
            B = images.size(0)
        hv = hv.reshape([B, -1, self.embed_dim])
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)
        hv_mask = get_hv_mask(hv)
        # 2) 视觉 CausalEncoder，得到 fl / fg
        hv, mediator = self.encoder(hv, mask=hv_mask,
                                    pos=self.vis_embed.pos_embed,
                                    proj=False)
        fl = mediator.get("local")
        fg = mediator.get("global")

        # 3) LGFM + pooling → X_path / X_noise
        mv_tokens = self.lgfm(fl, fg)  # [B,K,D]
        X_path = mv_tokens.mean(dim=1)  # [B,D]
        X_path = X_path.detach().requires_grad_(True)
        X_noise = fg.mean(dim=1)  # 仅为了接口完整性，这里可不必用

        # 4) 构造 F_path（同训练）
        B_l, K_l, D_l = fl.size()
        H, W = 1, K_l
        F_path = fl.transpose(1, 2).reshape(B_l, D_l, H, W)
        F_path = F.interpolate(F_path, size=(32, 32),
                               mode="bilinear", align_corners=False)
        # 5) Z：和训练一致，如果有 targets 就用 GT；否则全零
        if targets is not None:
            Z = self._get_Z(targets, mode='train', device=device)
        else:
            Z = torch.zeros(B, self.z_dim, device=device)
        # 6) seed + CEM
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            Y_seed = self.seed_mlp(X_path).unsqueeze(1)  # [B,1,D]
            cem_out = self.cem(X_path, Y_seed, Z=Z)
            S_g = cem_out.get("S_g_CEM", None)
        if isinstance(S_g, torch.Tensor):
            S_g = S_g.detach()
        mask_dummy = torch.ones(B, 1, 1, 1, device=device)

        return {
            "S_g_CEM": S_g,
            "mask_anat": mask_dummy,
            "X_path": X_path.detach()
        }

