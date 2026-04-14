import argparse
import csv
import json
import os
from typing import Iterable, List


def extract_reports(
    annotation_path: str,
    text_key: str,
    splits: Iterable[str],
) -> List[str]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reports: List[str] = []
    for split in splits:
        split_data = data.get(split, [])
        if not isinstance(split_data, list):
            continue
        for sample in split_data:
            if not isinstance(sample, dict):
                continue
            if text_key not in sample:
                raise KeyError(
                    f"样本缺少字段 '{text_key}'，请检查 annotation 结构。",
                )
            text = sample.get(text_key) or ""
            reports.append(str(text).replace("\r\n", " ").replace("\n", " ").strip())
    return reports


def main():
    parser = argparse.ArgumentParser(description="从 annotation.json 导出报告文本到单列 CSV。")
    parser.add_argument("--ann_path", required=True, help="原始 annotation.json 路径")
    parser.add_argument("--output_csv", required=True, help="输出 CSV 路径（无表头）")
    parser.add_argument(
        "--text_key",
        default="report",
        help="报告文本对应的字段名（默认: report）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="需要导出的数据划分，默认导出 train/val/test",
    )
    args = parser.parse_args()

    reports = extract_reports(
        annotation_path=args.ann_path,
        text_key=args.text_key,
        splits=args.splits,
    )

    if not reports:
        raise RuntimeError("未在指定划分中找到任何报告，请检查输入参数。")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for report in reports:
            writer.writerow([report])

    print(f"[Export] 已导出 {len(reports)} 条报告到 {args.output_csv}")


if __name__ == "__main__":
    main()


