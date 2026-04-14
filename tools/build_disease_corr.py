import argparse
import json
import os
from typing import Iterable, List

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def extract_label_lists(annotation: dict, split_keys: Iterable[str]) -> List[List[str]]:
    label_lists: List[List[str]] = []
    for split in split_keys:
        if split not in annotation or not isinstance(annotation[split], list):
            continue
        for sample in annotation[split]:
            labels = sample.get("labels")
            if not labels:
                continue
            if isinstance(labels, list):
                label_lists.append([str(label) for label in labels])
    return label_lists


def build_disease_corr(ann_path: str, output_path: str, split_keys: Iterable[str] = ("train", "val", "test")) -> None:
    with open(ann_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    label_lists = extract_label_lists(annotation, split_keys)
    if not label_lists:
        raise RuntimeError("未在标注文件中找到 'labels' 字段，请先运行 CheXbert 标签提取脚本。")

    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(label_lists)
    if label_matrix.size == 0:
        raise RuntimeError("标签矩阵为空，无法计算相关系数。")

    corr_matrix = np.corrcoef(label_matrix.T).astype(np.float32)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, corr_matrix)
    label_names_path = os.path.splitext(output_path)[0] + "_labels.json"
    with open(label_names_path, "w", encoding="utf-8") as f:
        json.dump({"labels": mlb.classes_.tolist()}, f, ensure_ascii=False, indent=2)

    print(f"[DiseaseCorr] 相关矩阵已保存至 {output_path} ，形状 {corr_matrix.shape}")
    print(f"[DiseaseCorr] 标签列表已保存至 {label_names_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于标注文件计算疾病共现相关矩阵")
    parser.add_argument("--ann_path", required=True, help="annotation.json 路径（需包含 labels 字段）")
    parser.add_argument("--output_path", required=True, help="输出 .npy 路径，例如 pretrain/disease_corr_iu_xray.npy")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_disease_corr(ann_path=args.ann_path, output_path=args.output_path)

