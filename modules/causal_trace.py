import torch
import torch.nn.functional as F

class CausalTraceAndResample:
    """
    因果回溯 + 重采样（推理阶段使用为主）
    - 阈值: hi(>=) 直接输出; lo(<=) 触发重采样; 中间区间仅插入不确定性提示
    - 默认只做“解码级重采样”，可选开启“特征级重采样”（需要传入 encode_fn）
    依赖于模型侧已缓存:
      - model._last_lambda_c: [B] or scalar tensor
      - model._last_H_chi: [B, 32, 32] (或同等尺度热图)
      - model.tokenizer.decode_batch(...)
      - model.beam_search.load_model(model.sample_forward, model.logit); .sample_beam(hv); .clean_model()
    """
    def __init__(self, hi: float = 0.8, lo: float = 0.4, p_top: float = 0.95,
                 beam_scale: float = 1.5, feature_resample: bool = True, resample_tries: int = 2):
        self.hi = float(hi);
        self.lo = float(lo);
        self.p_top = float(p_top)
        self.beam_scale = float(beam_scale)
        self.feature_resample = bool(feature_resample)
        self.resample_tries = int(resample_tries)

        # 轻量图像操作：模糊/反锐化 + 掩膜融合（CHI 的 top-p 区域）

    @staticmethod
    def _avg_blur(img, k=5):
        if k <= 1: return img
        pad = k // 2
        w = torch.ones(3, 1, k, k, device=img.device, dtype=img.dtype) / (k * k)
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        out = torch.zeros_like(img)
        for c in range(3):
            out[:, c:c + 1] = F.conv2d(img_pad[:, c:c + 1], w[c:c + 1], groups=1)
        return out

    @staticmethod
    def _unsharp(img, k=5, amount=0.75):
        blur = CausalTraceAndResample._avg_blur(img, k=k)
        return torch.clamp(img + amount * (img - blur), 0.0, 1.0)

    @staticmethod
    def _mask_blend(src, dst, mask):
        if mask.dim() == 3: mask = mask.unsqueeze(1)  # [B,1,H,W]
        return src * (1 - mask) + dst * mask

    @staticmethod
    def _half_split(x: torch.Tensor, dim: int = -1):
        n = x.size(dim)
        left = x.narrow(dim, 0, n // 2)
        right = x.narrow(dim, n // 2, n - n // 2)
        return left, right

    def _top_mask(self, heat, p_top=0.95):
        # 按 batch 做阈值：选取 top-p 的像素为 1，其余 0
        B = heat.size(0)
        flat = heat.flatten(1)
        k = (flat.size(1) * (1 - p_top))
        k = max(1, int(k))
        th = torch.topk(flat, k, dim=1).values.min(dim=1).values
        return (heat >= th.view(B, 1, 1)).float()

    @torch.no_grad()
    def _feature_resample_once(self, model, images, M_anat, H_chi):
        """支持 [B,3,H,W] 或 [B,V,3,H,W]，返回新 images 及 Omega"""
        need_reshape = (images.dim() == 5)
        if need_reshape:
            B, V, C, H, W = images.shape
            img_bv = images.view(B * V, C, H, W)
        else:
            B, C, H, W = images.shape
            V = 1
            img_bv = images

        Omega = self._top_mask(
            F.interpolate(H_chi.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                          ).squeeze(1), self.p_top)  # [B,H,W]

        ma = (M_anat > 0.5).float().squeeze(1)  # [B,H,W]
        outside = (Omega * (1 - ma)).sum((1, 2))
        inside = (Omega * ma).sum((1, 2))
        cause_noise = (outside > inside).view(B, 1, 1, 1)  # True=噪声, False=病灶模糊

        img0 = images.float()
        if img0.max() > 1.5: img0 = img0 / 255.0
        if need_reshape:
            img0_bv = img0.view(B * V, C, H, W)
            Omega_bv = Omega.repeat_interleave(V, dim=0)
        else:
            img0_bv = img0
            Omega_bv = Omega

        proc_noise = self._mask_blend(img0_bv, self._avg_blur(img0_bv, k=5), Omega_bv)
        proc_patho = self._mask_blend(img0_bv, self._unsharp(img0_bv, k=5, amount=0.9), Omega_bv)
        mixed_bv = torch.where(cause_noise.repeat_interleave(V, 0), proc_noise, proc_patho)
        mixed = mixed_bv.view(B, V, C, H, W) if need_reshape else mixed_bv
        return mixed.clamp(0, 1), Omega

    def decode_with_current_settings(self, model, hv, hv_mask=None):
        """
        使用当前模型的 beam_search 设置进行一次解码。
        需要模型提供:
          - model.sample_forward, model._masked_logit
          - model.beam_search.load_model(...), .sample_beam(hv, hv_mask), .clean_model()
          - model.tokenizer.decode_batch(...)
        """
        model.beam_search.load_model(model.sample_forward, model._masked_logit)

        # —— 将任意形状的 hv_mask 规范化 —— #
        # BeamSearch.sample_beam 期望 2D [B, S]；内部会 .unsqueeze(-2) -> [B, 1, S]
        # MultiHeadedAttention.forward 再 .unsqueeze(1) -> [B, 1, 1, S]
        if hv_mask is not None:
            if hv_mask.dim() == 4:  # [B,1,1,S] -> [B,S]
                hv_mask = hv_mask.squeeze(1).squeeze(1)
            elif hv_mask.dim() == 3:  # [B,1,S] -> [B,S]
                hv_mask = hv_mask.squeeze(1)
            elif hv_mask.dim() == 2:  # [B,S]
                pass
            else:
                hv_mask = None
            if hv_mask is not None and hv_mask.dtype not in (torch.long, torch.bool):
                hv_mask = hv_mask.long()

        # —— 安全缺省：没给就构造全 1 的 2D 掩码 —— #
        if hv_mask is None:
            B, S, _ = hv.size()
            hv_mask = torch.ones(B, S, dtype=torch.long, device=hv.device)

        # —— 解码：beam 优先；beam_size<=1 时走 greedy —— #
        if int(getattr(model.beam_search, "beam_size", 1)) <= 1:
            hv_mask_greedy = hv_mask.unsqueeze(-2)  # [B,S] -> [B,1,S]
            ids, _ = model._greedy_decode(hv, hv_mask_greedy)
        else:
            ids, _ = model.beam_search.sample_beam(hv, hv_mask)  # 2D [B,S]

        model.beam_search.clean_model()

        text = ""
        try:
            text = model.tokenizer.decode_batch(ids.detach().cpu().numpy())[0]
        except Exception:
            pass
        return ids, text

    def build_warning(self, H_chi, M_anat):
        B, Hh, Ww = H_chi.shape
        H_u = F.interpolate(H_chi.unsqueeze(1), size=(512, 512), mode="bilinear",
                            align_corners=False).squeeze(1)
        Omega = self._top_mask(H_u, self.p_top)  # [B,512,512]
        ma = (M_anat > 0.5).float().squeeze(1)  # [B,512,512]
        Om_L, Om_R = self._half_split(Omega, -1)
        ma_L, ma_R = self._half_split(ma, -1)
        left_e = (Om_L * ma_L).sum((1, 2))
        right_e = (Om_R * ma_R).sum((1, 2))
        warns = [("右肺" if right_e[i] >= left_e[i] else "左肺") + "存在因果空洞高风险区域，建议结合临床与多视角复核。"
                 for i in range(B)]
        return warns, Omega

    def apply(self, model, images, hv, hv_mask=None, use_resample=True, encode_fn=None, max_feature_resamples=0):
        device = hv.device
        B = hv.size(0)

        # 1) 初次解码
        base_ids, base_text = self.decode_with_current_settings(model, hv, hv_mask)
        lam = getattr(model, "_last_lambda_c", None)
        lam = torch.ones(B, device=device) if lam is None else lam
        lam_s = float(lam.mean().detach().item()) if isinstance(lam, torch.Tensor) else float(lam)

        # 2) 风险提示文本
        warn_txt, Omega = "", None
        Hchi = getattr(model, "_last_H_chi", None)
        Manat = getattr(model, "_last_M_anat", None)
        if isinstance(Hchi, torch.Tensor) and Hchi.dim() == 3 and isinstance(Manat, torch.Tensor):
            warns, Omega = self.build_warning(Hchi, Manat)
            warn_txt = warns[0] if len(warns) else ""

        # 3) 置信度门控
        if lam_s >= self.hi:
            return base_ids, base_text.strip(), {"strategy": "base", "lambda": lam_s, "warning": ""}
        if lam_s >= self.lo:
            out = base_text.strip()
            if warn_txt: out += (" " if out.endswith(".") else ". ") + f"[不确定性提示] {warn_txt}"
            return base_ids, out, {"strategy": "warn_only", "lambda": lam_s, "warning": warn_txt}

        # 4) 低置信：优先特征级重采样（可配次数）
        tries = int(max_feature_resamples or self.resample_tries)
        best_ids, best_txt, lam_best = base_ids, base_text, lam_s

        if use_resample and self.feature_resample and tries > 0 and Omega is not None:
            img_cur = images
            for _ in range(tries):
                img_rs, _ = self._feature_resample_once(model, img_cur, Manat, Hchi)
                # 重新编码 + DyCE + CHI
                if img_rs.dim() == 5:
                    Bv, V, C, H, W = img_rs.shape
                    hv_new = model.vis_embed(img_rs.reshape(Bv * V, C, H, W)).reshape(Bv, -1, model.embed_dim)
                else:
                    hv_new = model.vis_embed(img_rs).reshape(B, -1, model.embed_dim)

                Xp, Xn, Fp, Ma2 = model.feat(img_rs)
                with torch.set_grad_enabled(True):
                    f_gap = F.adaptive_avg_pool2d(Fp, (1, 1)).flatten(1)
                    Y_seed = model.seed_mlp(f_gap).unsqueeze(1)
                    X_fin, Y_fin, lamb_k, _ = model.dyce(Xp, Y_seed, Fp, Xn)

                chi_out = model.chi(Fp, Y_fin, Xn)
                model._last_lambda_c = chi_out["lambda_c"].detach()
                model._last_H_chi = chi_out["H_chi"].detach()
                model._last_M_anat = Ma2.detach()
                lam_try = float(model._last_lambda_c.mean().item())

                ids_try, txt_try = self.decode_with_current_settings(model, hv_new, hv_mask)
                if len(str(txt_try).split()) >= len(str(best_txt).split()):
                    best_ids, best_txt, lam_best = ids_try, txt_try, lam_try
                if lam_try >= self.hi:
                    out_txt = str(txt_try).strip()
                    if warn_txt: out_txt += (" " if out_txt.endswith(
                        ".") else ". ") + f"[重采样提醒] 已进行特征级重采样。{warn_txt}"
                    return ids_try, out_txt, {"strategy": "feature_resample", "lambda": lam_try, "warning": warn_txt}
                img_cur = img_rs

            out_txt = str(best_txt).strip()
            if warn_txt: out_txt += (" " if out_txt.endswith(
                ".") else ". ") + f"[重采样提醒] 已进行特征级重采样。{warn_txt}"
            return best_ids, out_txt, {"strategy": "feature_resample", "lambda": lam_best, "warning": warn_txt}

        # 5) 兜底：解码级重采样（增大 beam）
        best_ids, best_txt = base_ids, base_text
        if hasattr(model, "beam_search"):
            orig_beam = getattr(model.beam_search, "beam_size", 3)
            try:
                model.beam_search.beam_size = max(5, int(orig_beam * self.beam_scale))
                ids2, txt2 = self.decode_with_current_settings(model, hv, hv_mask)
                if len(str(txt2).split()) >= len(str(best_txt).split()):
                    best_ids, best_txt = ids2, txt2
            finally:
                model.beam_search.beam_size = orig_beam

        out = str(best_txt).strip()
        if warn_txt: out += (" " if out.endswith(
            ".") else ". ") + f"[重采样提醒] 该报告置信度较低，已触发策略性重采样。{warn_txt}"
        return best_ids, out, {"strategy": "resample", "lambda": lam_s, "warning": warn_txt}
