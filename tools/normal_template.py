"""
tool/normal_template.py
离线构建群体“正常原型” θ_norm：
- 规则筛选报告中的 "No Finding" 样本
- 前向抽取 X_path:[512]，做均值
- 保存为 pretrain/normal_template.np
"""
# utils/normal_template.py
# 多数据集通用：筛选 "No Finding" 样本 → 抽取 X_path:[512] → 均值 → 保存 θ_norm

# ---- 让 python utils/normal_template.py 可直接运行（保证能 import modules.*） ----
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# -------------------------------------------------------------------------------

import re
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modules.feature_disentanglement import FeatureDisentanglement


# ======================= 数据集路径推断 =======================
def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None

def _guess_base_dir(dataset_name: str) -> str:
    dn = dataset_name.strip()
    variants = {dn, dn.replace("_", "-"), dn.replace("-", "_")}
    if dn.lower() in {"mimic", "mimiccxr", "mimic-cxr", "mimic_cxr"}:
        variants |= {"mimic-cxr", "mimic_cxr"}
    if "iu" in dn.lower():
        variants |= {"iu_xray", "iu-xray", "IU_Xray"}
    candidates = [os.path.join("data", v) for v in variants]
    candidates.append(os.path.join("data", dn))
    base = _first_existing(candidates)
    return base or os.path.join("data", dn)

def _guess_image_dir(base_dir: str) -> str:
    img_dir = os.path.join(base_dir, "images")
    return img_dir if os.path.isdir(img_dir) else base_dir

def _guess_ann_path(base_dir: str) -> str:
    pref = os.path.join(base_dir, "annotation.json")
    if os.path.isfile(pref):
        return pref
    cand = []
    for fn in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
        if fn.lower().endswith(".json") and ("ann" in fn.lower() or "annotation" in fn.lower()):
            cand.append(os.path.join(base_dir, fn))
    if len(cand) == 1:
        return cand[0]
    for c in cand:
        if "annotation" in os.path.basename(c).lower():
            return c
    return cand[0] if cand else pref


# ======================= 文本预处理 =======================
def sanitize_text(t: str) -> str:
    t = t.lower()
    t = t.replace("-"," ").replace("_"," ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ======================= 放宽/补充后的 No Finding 规则 =======================
def build_is_no_finding_fn(pos_min: int = 1, loose_mode: bool = False) -> callable:
    POS_PATTERNS = [
        r'\bno (acute )?cardiopulmonary (disease|process|abnormalit(?:y|ies))\b',
        r'\bno acute (cardiopulmonary )?(disease|abnormalit(?:y|ies)|process)\b',
        r'\bno active (cardiopulmonary )?disease\b',
        r'\bno (focal )?(airspace|alveolar) (disease|opacity|opacities|consolidation)\b',
        r'\bno (focal )?consolidation\b',
        r'\bno focal (opacity|opacities)\b',
        r'\bno pneumothorax\b',
        r'\bno pleural (effusion|effusions|fluid)\b',
        r'\bno edema\b',
        r'\bno pneumonia\b',
        r'\bno acute osseous abnormalit(?:y|ies)\b',
        r'\bno displaced (rib )?fracture\b',
        r'\bcardiomediastinal (?:silhouette )?(?:is )?(?:within )?normal (?:limits)?\b',
        r'\bheart size (?:is )?normal\b',
        r'\bclear lungs?\b',
        r'\bnormal chest(?: radiograph| x?ray)?\b',
        r'\bunchanged (when compared with|from) prior\b',
        r'\bno interval change\b',
        r'\bno acute intrathoracic process\b',
        r'\bno radiographic evidence of (?:acute )?cardiopulmonary (?:disease|process|abnormalit(?:y|ies))\b',
        r'\bno acute cardiopulmonary abnormalit(?:y|ies)\b',
    ]
    NEG_TERMS = [
        r'atelectasis', r'pneumothorax', r'consolidation', r'opacity', r'opacities',
        r'effusion', r'edema', r'infiltrate', r'fracture', r'pneumonia', r'cardiomegaly',
        r'enlarg(?:ed|ement)', r'pleural thickening', r'nodule', r'mass', r'lesion',
        r'hyperinflation', r'tube|line|lead|pacer|support device|catheter|ett',
        r'hemothorax', r'emphysema', r'fibrosis', r'scar', r'atelectatic',
        r'congest(ion|ive)', r'concerning for', r'suspicious for', r'worrisome',
        r'cannot exclude', r'cannot be excluded',
    ]
    POS_RE = [re.compile(p, re.IGNORECASE) for p in POS_PATTERNS]
    NEG_RE = [re.compile(p, re.IGNORECASE) for p in NEG_TERMS]

    def is_nf(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        t = sanitize_text(text)
        if loose_mode:
            # 松弛模式：至少一个关键正向短语，且不含核心负项
            loose_pos = [
                r'\bno acute cardiopulmonary\b',
                r'\bno acute disease\b',
                r'\bclear lungs?\b',
                r'\bnormal chest\b',
                r'\bno focal consolidation\b',
            ]
            LP = [re.compile(p) for p in loose_pos]
            if not any(p.search(t) for p in LP):
                return False
            core_neg = [
                r'atelectasis', r'consolidation', r'effusion', r'pneumothorax', r'pneumonia',
                r'cardiomegaly', r'\benlarg(?:ed|ement)\b', r'enlarged pulmonary arter(?:y|ies)',
                r'pulmonary artery (?:enlarged|enlargement)',
                r'hernia', r'granuloma', r'nodule', r'mass', r'lesion',
                r'scar|scarring|fibrosis|emphysema',
                r'sternotomy', r'prosthetic', r'prosthesis', r'implant', r'device', r'pac(?:er|emaker)',
                r'line|tube|catheter|ett',
            ]

            if any(re.search(p, t) for p in core_neg):
                return False
            return True

        pos_hits = sum(bool(r.search(t)) for r in POS_RE)
        neg_hits = sum(bool(r.search(t)) for r in NEG_RE)
        return (pos_hits >= pos_min) and (neg_hits == 0)

    return is_nf


# ======================= 注解加载 & 图像读取 =======================
def extract_image_path_and_text(example: Dict[str, Any], image_dir: str) -> Optional[Tuple[str, str]]:
    img_rel = None
    for k in ['image_path', 'img_path', 'path', 'image', 'file_name']:
        if k in example and isinstance(example[k], (str, list)):
            v = example[k]
            if isinstance(v, list) and len(v) > 0:
                img_rel = v[0]
            elif isinstance(v, str):
                img_rel = v
            break
    text = None
    for key in ['report', 'report_impression', 'impression', 'findings', 'caption', 'text']:
        if key in example and isinstance(example[key], str) and len(example[key]) > 0:
            text = example[key]
            break
    if img_rel is None or text is None:
        return None
    return os.path.join(image_dir, img_rel), text

def load_annotations(ann_path: str, split: str = "train") -> List[Dict[str, Any]]:
    data = json.loads(open(ann_path, 'r', encoding='utf-8').read())
    if isinstance(data, dict):
        if split == "all":
            merged = []
            for v in data.values():
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                    merged.extend(v)
            if len(merged) > 0:
                return merged
        if split in data and isinstance(data[split], list):
            return data[split]
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return v
        raise ValueError(f"Unrecognized annotation dict schema in {ann_path}")
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported annotation type: {type(data)}")

def pil_to_tensor_512(im: Image.Image) -> torch.Tensor:
    im = im.convert('RGB').resize((512, 512))
    arr = np.asarray(im).astype('float32') / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


# ======================= 主流程 =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_name', type=str, required=True)
    ap.add_argument('--image_dir', type=str, default="")
    ap.add_argument('--ann_path',  type=str, default="")
    ap.add_argument('--save_path', type=str, default="")
    ap.add_argument('--unet_ckpt', type=str, default='pretrain/unet_resnet_medical.pth')
    ap.add_argument('--vit_ckpt',  type=str, default='pretrain/vit_base_patch16_224_in21k.pth')
    ap.add_argument('--freeze_unet', type=lambda s: s.lower() in ['true','1','yes'], default=True)
    ap.add_argument('--device',      type=str, default='cuda:0')
    ap.add_argument('--max_samples', type=int, default=0)
    ap.add_argument('--split',       type=str, default="train", choices=["train","val","test","all"])
    ap.add_argument('--pos_hits_min',type=int, default=1, help="正向短语最少命中数（越大越保守）")
    ap.add_argument('--loose_mode',  action='store_true', help="使用更松弛的 NF 判定")
    ap.add_argument('--debug_k',     type=int, default=0, help="打印前 K 条匹配/未匹配样例用于调试")
    args = ap.parse_args()

    base_dir = _guess_base_dir(args.dataset_name)
    image_dir = args.image_dir or _guess_image_dir(base_dir)
    ann_path  = args.ann_path  or _guess_ann_path(base_dir)
    save_path = args.save_path or os.path.join("pretrain", f"normal_template_{args.dataset_name}.npy")

    print(f"[CFG] dataset_name   = {args.dataset_name}")
    print(f"[CFG] base_dir       = {base_dir}")
    print(f"[CFG] image_dir      = {image_dir}   (exists={os.path.isdir(image_dir)})")
    print(f"[CFG] ann_path       = {ann_path}    (exists={os.path.isfile(ann_path)})")
    print(f"[CFG] save_path      = {save_path}")
    print(f"[CFG] split          = {args.split}")
    print(f"[CFG] ckpt(unet/vit) = {args.unet_ckpt} / {args.vit_ckpt}")
    print(f"[CFG] pos_hits_min   = {args.pos_hits_min}, loose_mode={args.loose_mode}")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"[E] image_dir not found: {image_dir}")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"[E] ann_path not found: {ann_path}")

    is_nf = build_is_no_finding_fn(pos_min=args.pos_hits_min, loose_mode=args.loose_mode)

    examples = load_annotations(ann_path, split=args.split)
    print(f"[INFO] Loaded {len(examples)} examples from {ann_path} (split={args.split})")

    pos_samples, neg_samples = [], []
    candidates: List[Tuple[str, str]] = []
    for ex in examples:
        item = extract_image_path_and_text(ex, image_dir)
        if item is None:
            continue
        img_abs, text = item
        if not os.path.isfile(img_abs):
            continue
        if is_nf(text):
            candidates.append((img_abs, text))
            if len(pos_samples) < args.debug_k:
                pos_samples.append(text)
        elif len(neg_samples) < args.debug_k:
            neg_samples.append(text)

    print(f"[INFO] Found {len(candidates)} 'No Finding' candidates via rule-based filter.")

    if args.debug_k > 0:
        print("\n[DEBUG] --- POSITIVE SAMPLES ---")
        for i, s in enumerate(pos_samples, 1):
            print(f"[+{i}] {s[:300]}")
        print("\n[DEBUG] --- NEGATIVE SAMPLES ---")
        for i, s in enumerate(neg_samples, 1):
            print(f"[-{i}] {s[:300]}")
        print("[DEBUG] --- END ---\n")

    if len(candidates) == 0:
        raise RuntimeError("No 'No Finding' examples found. Try --pos_hits_min 1 --loose_mode or check text fields/paths.")

    if args.max_samples and args.max_samples > 0:
        candidates = candidates[:args.max_samples]

    device = torch.device(args.device if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
    feat = FeatureDisentanglement(
        unet_ckpt=args.unet_ckpt,
        vit_ckpt=args.vit_ckpt,
        freeze_unet=args.freeze_unet,
        project_dim=512,
    ).to(device).eval()

    sum_x = torch.zeros(512, dtype=torch.float32, device=device)
    count = 0
    with torch.no_grad():
        for i, (img_abs, _) in enumerate(candidates, 1):
            try:
                im = Image.open(img_abs)
                x = pil_to_tensor_512(im).unsqueeze(0).to(device)  # [1,3,512,512]
                x_path, x_noise, f_path, m_anat = feat(x)          # x_path: [1,512]
                sum_x += x_path.squeeze(0)
                count += 1
            except Exception as e:
                print(f"[WARN] Skip {img_abs}: {e}")
                continue
            if i % 50 == 0:
                print(f"[INFO] processed {i}/{len(candidates)}")

    if count == 0:
        raise RuntimeError("No images processed successfully.")

    mean_x = (sum_x / count).detach().cpu().numpy().astype('float32')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, mean_x)
    print(f"[OK] Saved normal_template to: {save_path}  (vector shape: {mean_x.shape}, count={count})")

    meta = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "count": int(count),
        "x_dim": 512,
        "image_dir": image_dir,
        "ann_path": ann_path,
        "freeze_unet": bool(args.freeze_unet),
        "vit_ckpt": args.vit_ckpt,
        "pos_hits_min": int(args.pos_hits_min),
        "loose_mode": bool(args.loose_mode),
        "patterns_version": 3,
    }
    meta_path = os.path.splitext(save_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
