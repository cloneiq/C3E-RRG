import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from utils.tokenizers_utils import Tokenizer


def infer_dataset_name(path: str, explicit: str = None) -> str:
    if explicit:
        return explicit
    lower = path.lower()
    if "iu_xray" in lower:
        return "iu_xray"
    if "mimic" in lower:
        return "mimic_cxr"
    raise ValueError("无法从路径推断数据集名称，请通过 --dataset 显式指定（支持 iu_xray / mimic_cxr）。")


def load_vocab_from_tokenizer(tokenizer: Tokenizer) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从 tokenizer 直接获取词汇表"""
    token2idx = tokenizer.token2idx
    idx2token = tokenizer.idx2token
    return token2idx, idx2token


def load_vocab(vocab_path: str = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从文件加载词汇表（如果提供）"""
    if vocab_path is None or not os.path.exists(vocab_path):
        return None, None
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token2idx = vocab.get("token2idx")
    idx2token = vocab.get("idx2token")

    if token2idx is None and idx2token is None:
        return None, None

    if token2idx is None and idx2token is not None:
        token2idx = {token: int(idx) for idx, token in idx2token.items()}
    if idx2token is None and token2idx is not None:
        idx2token = {int(idx): token for token, idx in token2idx.items()}
    else:
        idx2token = {int(k): v for k, v in idx2token.items()}

    return token2idx, idx2token


def tokenize_reports(annotation: dict, tokenizer: Tokenizer, split_keys: Iterable[str]) -> List[List[str]]:
    tokenized_reports: List[List[str]] = []
    for split in split_keys:
        if split not in annotation or not isinstance(annotation[split], list):
            continue
        for sample in annotation[split]:
            report = sample.get("report", "")
            if not isinstance(report, str):
                continue
            cleaned = tokenizer.clean_report(report)
            tokens = cleaned.split()
            if tokens:
                tokenized_reports.append(tokens)
    return tokenized_reports


def compute_pmi(
    documents: List[List[str]],
    token2idx: Dict[str, int],
    min_count: int = 5,
    window_size: int = 5,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    计算 PMI 矩阵
    
    注意：tokenizer 的索引从 1 开始（0 是 <pad>），但矩阵索引从 0 开始
    """
    vocab_indices = {token: idx for token, idx in token2idx.items()}
    # tokenizer 索引从 1 开始，最大索引就是词汇表大小（不包括 pad=0）
    max_idx = max(vocab_indices.values()) if vocab_indices else 0
    # 矩阵大小应该是 max_idx + 1（包括索引 0，虽然 pad 不参与 PMI 计算）
    vocab_size = max_idx + 1
    # 索引偏移：tokenizer 索引从 1 开始，矩阵从 0 开始
    index_shift = 1 if 0 not in vocab_indices.values() else 0

    word_count: Counter = Counter()
    cooc_count: defaultdict = defaultdict(int)

    for tokens in documents:
        length = len(tokens)
        for i, w1 in enumerate(tokens):
            if w1 not in vocab_indices:
                continue
            word_count[w1] += 1
            left = max(0, i - window_size)
            right = min(length, i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                w2 = tokens[j]
                if w2 not in vocab_indices:
                    continue
                cooc_count[(w1, w2)] += 1

    valid_words = {w for w, c in word_count.items() if c >= min_count}
    total_tokens = sum(word_count.values())
    total_pairs = sum(cooc_count.values())

    if total_tokens == 0 or total_pairs == 0:
        raise RuntimeError("未找到有效的词共现对，请检查数据或降低 min_count 阈值。")

    matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    for (w1, w2), c in cooc_count.items():
        if w1 not in valid_words or w2 not in valid_words:
            continue
        i = vocab_indices[w1] - index_shift
        j = vocab_indices[w2] - index_shift
        if i < 0 or j < 0 or i >= vocab_size or j >= vocab_size:
            continue
        p_w1 = word_count[w1] / total_tokens
        p_w2 = word_count[w2] / total_tokens
        p_w1w2 = c / (total_pairs + eps)
        denom = p_w1 * p_w2 + eps
        value = math.log(p_w1w2 / denom + eps)
        matrix[i, j] = max(0.0, value)  # PPMI

    return matrix


def build_pmi_matrix(
    ann_path: str,
    output_path: str,
    vocab_path: str = None,
    dataset_name: str = None,
    min_count: int = 5,
    window_size: int = 5,
    threshold: int = None,
) -> None:
    """
    构建 PMI 矩阵
    
    Args:
        ann_path: annotation.json 路径
        output_path: 输出文件路径
        vocab_path: 可选的词汇表 JSON 路径（如果提供则使用，否则从 tokenizer 自动生成）
        dataset_name: 数据集名称（iu_xray 或 mimic_cxr）
        min_count: 最小词频阈值
        window_size: PMI 统计窗口大小
        threshold: tokenizer 的词频阈值（仅在自动生成词汇表时使用）
    """
    dataset_name = infer_dataset_name(ann_path, dataset_name)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # 根据数据集设置默认 threshold
    if threshold is None:
        threshold = 3 if dataset_name == "iu_xray" else 10
    
    # 创建 tokenizer（会自动生成词汇表）
    tokenizer_args = {
        "ann_path": ann_path,
        "threshold": threshold,
        "dataset_name": dataset_name,
    }
    tokenizer = Tokenizer(tokenizer_args)

    # 优先使用文件中的词汇表，否则使用 tokenizer 生成的
    token2idx_file, idx2token_file = load_vocab(vocab_path)
    if token2idx_file is not None and idx2token_file is not None:
        print(f"[PMI] 使用提供的词汇表文件: {vocab_path}")
        token2idx, idx2token = token2idx_file, idx2token_file
    else:
        print(f"[PMI] 从 tokenizer 自动生成词汇表（词频阈值={threshold}）")
        token2idx, idx2token = load_vocab_from_tokenizer(tokenizer)
        # 可选：保存词汇表到文件
        if vocab_path:
            vocab_dir = os.path.dirname(vocab_path) or "."
            os.makedirs(vocab_dir, exist_ok=True)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump({"token2idx": token2idx, "idx2token": idx2token}, f, ensure_ascii=False, indent=2)
            print(f"[PMI] 词汇表已保存至 {vocab_path}")

    documents = tokenize_reports(annotation, tokenizer, split_keys=["train", "val", "test"])
    if not documents:
        raise RuntimeError("未从标注文件中提取到报告文本。")

    print(f"[PMI] 处理了 {len(documents)} 份报告，词汇表大小: {len(token2idx)}")

    matrix = compute_pmi(
        documents=documents,
        token2idx=token2idx,
        min_count=min_count,
        window_size=window_size,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {
            "pmi": matrix,
            "token2idx": token2idx,
            "idx2token": idx2token,
            "min_count": min_count,
            "window_size": window_size,
        },
        output_path,
    )
    print(f"[PMI] 矩阵已保存至 {output_path}，形状 {tuple(matrix.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 PMI / PPMI 语言先验矩阵")
    parser.add_argument("--ann_path", required=True, help="annotation.json 路径")
    parser.add_argument("--output_path", required=True, help="输出文件路径，例如 pretrain/pmi_matrix_iu_xray.pt")
    parser.add_argument("--vocab_path", default=None, help="可选的词汇表 JSON 路径（如果提供则使用，否则从 tokenizer 自动生成）")
    parser.add_argument("--dataset", choices=["iu_xray", "mimic_cxr"], help="数据集名称，可选（会自动从路径推断）")
    parser.add_argument("--min_count", type=int, default=5, help="最小词频阈值（用于 PMI 计算）")
    parser.add_argument("--window_size", type=int, default=5, help="PMI 统计窗口大小")
    parser.add_argument("--threshold", type=int, default=None, help="Tokenizer 词频阈值（用于自动生成词汇表，不指定则根据数据集自动设置：iu_xray=3, mimic_cxr=10）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pmi_matrix(
        ann_path=args.ann_path,
        output_path=args.output_path,
        vocab_path=args.vocab_path,
        dataset_name=args.dataset,
        min_count=args.min_count,
        window_size=args.window_size,
        threshold=args.threshold,
    )

