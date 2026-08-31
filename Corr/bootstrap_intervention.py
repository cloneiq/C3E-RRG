import os
import argparse
import numpy as np
import pandas as pd


PAIR = ["Cardiomegaly", "Pleural Effusion"]


# ------------------------------------------------------------
# Label / metric utilities
# ------------------------------------------------------------
def to_binary_positive(series):
    """
    Match the convention used by the existing intervention evaluator:
    explicit positive labels are mapped to 1; all other values to 0.

    This function is intended for the already-generated CheXbert label CSVs.
    """
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    out = (num == 1).astype(int)

    text = s.astype(str).str.strip().str.lower()
    positive_tokens = {
        "positive", "pos", "present", "true", "yes", "1", "1.0"
    }
    out = ((out == 1) | text.isin(positive_tokens)).astype(int)
    return out


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / np.sum(w))


def weighted_corr(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    if len(x) == 0 or np.sum(w) <= 0:
        return np.nan

    w = w / np.sum(w)
    mx = np.sum(w * x)
    my = np.sum(w * y)

    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan

    return float(cov / np.sqrt(vx * vy))


def pair_micro_prf(y_true, y_pred, w):
    """
    Weighted micro P/R/F1 over the two target disease labels.
    y_true / y_pred: [N, 2]
    w: [N] sample weights
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    w2 = np.asarray(w, dtype=float).reshape(-1, 1)

    tp = float(np.sum(w2 * ((y_true == 1) & (y_pred == 1))))
    fp = float(np.sum(w2 * ((y_true == 0) & (y_pred == 1))))
    fn = float(np.sum(w2 * ((y_true == 1) & (y_pred == 0))))

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ------------------------------------------------------------
# File alignment
# ------------------------------------------------------------
def read_report(path):
    df = pd.read_csv(path)
    required = ["subject_id", "study_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")

    df = df.copy()
    df["subject_id"] = pd.to_numeric(
        df["subject_id"], errors="raise"
    ).astype("int64")
    df["study_id"] = pd.to_numeric(
        df["study_id"], errors="raise"
    ).astype("int64")
    return df


def attach_label_file(label_path, report_df, prefix):
    """
    Supports either:
      1) label CSV contains subject_id/study_id -> merge by IDs;
      2) label CSV contains only disease columns -> align by row order,
         requiring exactly the same number of rows as report.csv.
    """
    lab = pd.read_csv(label_path)

    missing = [c for c in PAIR if c not in lab.columns]
    if missing:
        raise KeyError(f"{label_path}: missing disease columns {missing}")

    if "study_id" in lab.columns:
        lab = lab.copy()
        lab["study_id"] = pd.to_numeric(
            lab["study_id"], errors="raise"
        ).astype("int64")

        if "subject_id" in lab.columns:
            lab["subject_id"] = pd.to_numeric(
                lab["subject_id"], errors="raise"
            ).astype("int64")
            keys = ["subject_id", "study_id"]
        else:
            keys = ["study_id"]

        merged = report_df[["subject_id", "study_id"]].merge(
            lab[keys + PAIR],
            on=keys,
            how="left",
            validate="many_to_one",
        )

        if merged[PAIR].isna().all(axis=1).any():
            n_bad = int(merged[PAIR].isna().all(axis=1).sum())
            raise RuntimeError(
                f"{label_path}: {n_bad} report rows failed ID matching."
            )
    else:
        if len(lab) != len(report_df):
            raise RuntimeError(
                f"{label_path}: {len(lab)} rows, but report.csv has "
                f"{len(report_df)} rows. Row-order alignment is unsafe."
            )

        merged = report_df[["subject_id", "study_id"]].copy()
        merged["Cardiomegaly"] = lab["Cardiomegaly"].values
        merged["Pleural Effusion"] = lab["Pleural Effusion"].values

    merged[f"{prefix}_C"] = to_binary_positive(
        merged["Cardiomegaly"]
    )
    merged[f"{prefix}_E"] = to_binary_positive(
        merged["Pleural Effusion"]
    )

    return merged[
        ["subject_id", "study_id", f"{prefix}_C", f"{prefix}_E"]
    ].copy()


def prepare_data(args):
    wo_report = read_report(args.woconf_report)
    fu_report = read_report(args.full_report)

    wo_keys = list(zip(wo_report["subject_id"], wo_report["study_id"]))
    fu_keys = list(zip(fu_report["subject_id"], fu_report["study_id"]))

    if wo_keys != fu_keys:
        raise RuntimeError(
            "The two report.csv files do not have identical "
            "(subject_id, study_id) sequences."
        )

    wo_gt = attach_label_file(
        args.woconf_gt_labels, wo_report, prefix="gt"
    )
    fu_gt = attach_label_file(
        args.full_gt_labels, fu_report, prefix="gt"
    )

    if not np.array_equal(
        wo_gt[["gt_C", "gt_E"]].to_numpy(),
        fu_gt[["gt_C", "gt_E"]].to_numpy(),
    ):
        raise RuntimeError(
            "The two GT label files yield different "
            "Cardiomegaly/Pleural Effusion labels."
        )

    wo_pred = attach_label_file(
        args.woconf_pred_labels, wo_report, prefix="wo"
    )
    fu_pred = attach_label_file(
        args.full_pred_labels, fu_report, prefix="fu"
    )

    data = wo_report[["subject_id", "study_id"]].copy()
    data["gt_C"] = wo_gt["gt_C"].to_numpy()
    data["gt_E"] = wo_gt["gt_E"].to_numpy()

    data["wo_C"] = wo_pred["wo_C"].to_numpy()
    data["wo_E"] = wo_pred["wo_E"].to_numpy()

    data["fu_C"] = fu_pred["fu_C"].to_numpy()
    data["fu_E"] = fu_pred["fu_E"].to_numpy()

    # Composite study-level cluster key.
    data["_cluster"] = (
        data["subject_id"].astype(str)
        + "_"
        + data["study_id"].astype(str)
    )

    return data


# ------------------------------------------------------------
# Intervention construction
# ------------------------------------------------------------
def build_decorrelation_weights(gt_c, gt_e):
    """
    Recompute the decorrelation intervention within the current
    bootstrap sample:

        Q(C,E) = P(C) P(E)

    Returns a sample-level importance weight vector and distribution info.
    """
    c = np.asarray(gt_c, dtype=int)
    e = np.asarray(gt_e, dtype=int)
    n = len(c)

    p_c1 = float(np.mean(c))
    p_e1 = float(np.mean(e))

    p_c = {0: 1.0 - p_c1, 1: p_c1}
    p_e = {0: 1.0 - p_e1, 1: p_e1}

    joint = {}
    for cv in [0, 1]:
        for ev in [0, 1]:
            joint[(cv, ev)] = int(np.sum((c == cv) & (e == ev)))

    if any(v == 0 for v in joint.values()):
        return None, None

    lookup = {}
    for cv in [0, 1]:
        for ev in [0, 1]:
            p_joint = joint[(cv, ev)] / n
            q_target = p_c[cv] * p_e[ev]
            lookup[(cv, ev)] = q_target / p_joint

    w = np.asarray(
        [lookup[(int(cv), int(ev))] for cv, ev in zip(c, e)],
        dtype=float,
    )
    w /= np.mean(w)

    info = {
        "c_prev_orig": p_c1,
        "e_prev_orig": p_e1,
        "c_prev_decor": weighted_mean(c, w),
        "e_prev_decor": weighted_mean(e, w),
        "gt_corr_orig": float(np.corrcoef(c, e)[0, 1]),
        "gt_corr_decor": weighted_corr(c, e, w),
    }
    return w, info


def evaluate_predictions(gt, pred, w_decor, info):
    y_true = np.asarray(gt, dtype=int)
    y_pred = np.asarray(pred, dtype=int)

    w_orig = np.ones(len(y_true), dtype=float)

    _, _, f1_orig = pair_micro_prf(
        y_true, y_pred, w_orig
    )
    _, _, f1_decor = pair_micro_prf(
        y_true, y_pred, w_decor
    )

    pred_corr_orig = weighted_corr(
        y_pred[:, 0], y_pred[:, 1], w_orig
    )
    pred_corr_decor = weighted_corr(
        y_pred[:, 0], y_pred[:, 1], w_decor
    )

    delta_f1 = abs(f1_decor - f1_orig)

    # Ground-truth decorrelated correlation is theoretically 0;
    # use the numerical value for a robust residual measure.
    residual_corr_decor = abs(
        pred_corr_decor - info["gt_corr_decor"]
    )

    return {
        "pair_f1_orig": f1_orig,
        "pair_f1_decor": f1_decor,
        "delta_f1": delta_f1,
        "pred_corr_orig": pred_corr_orig,
        "pred_corr_decor": pred_corr_decor,
        "residual_corr_decor": residual_corr_decor,
    }


def evaluate_sample(sample):
    gt = sample[["gt_C", "gt_E"]].to_numpy(dtype=int)

    w_decor, info = build_decorrelation_weights(
        sample["gt_C"].to_numpy(),
        sample["gt_E"].to_numpy(),
    )
    if w_decor is None:
        return None

    wo_pred = sample[["wo_C", "wo_E"]].to_numpy(dtype=int)
    fu_pred = sample[["fu_C", "fu_E"]].to_numpy(dtype=int)

    wo = evaluate_predictions(
        gt, wo_pred, w_decor, info
    )
    fu = evaluate_predictions(
        gt, fu_pred, w_decor, info
    )

    return {
        "n_rows": len(sample),
        "n_clusters": sample["_cluster"].nunique(),

        "gt_corr_orig": info["gt_corr_orig"],
        "gt_corr_decor": info["gt_corr_decor"],

        "wo_pair_f1_orig": wo["pair_f1_orig"],
        "wo_pair_f1_decor": wo["pair_f1_decor"],
        "wo_delta_f1": wo["delta_f1"],
        "wo_pred_corr_orig": wo["pred_corr_orig"],
        "wo_pred_corr_decor": wo["pred_corr_decor"],
        "wo_residual_corr_decor": wo["residual_corr_decor"],

        "fu_pair_f1_orig": fu["pair_f1_orig"],
        "fu_pair_f1_decor": fu["pair_f1_decor"],
        "fu_delta_f1": fu["delta_f1"],
        "fu_pred_corr_orig": fu["pred_corr_orig"],
        "fu_pred_corr_decor": fu["pred_corr_decor"],
        "fu_residual_corr_decor": fu["residual_corr_decor"],

        # Positive values favor the Full model.
        "paired_delta_f1_reduction": (
            wo["delta_f1"] - fu["delta_f1"]
        ),
        "paired_residual_corr_reduction": (
            wo["residual_corr_decor"]
            - fu["residual_corr_decor"]
        ),
    }


# ------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------
def percentile_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    low = float(np.percentile(values, 100 * alpha / 2))
    high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return low, high


def summarize_metric(name, point, boot_values):
    arr = np.asarray(boot_values, dtype=float)
    lo, hi = percentile_ci(arr, alpha=0.05)

    return {
        "metric": name,
        "point_estimate": float(point),
        "bootstrap_mean": float(np.mean(arr)),
        "bootstrap_std": float(np.std(arr, ddof=1)),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Study-level paired cluster bootstrap for the "
            "Cardiomegaly-Pleural Effusion intervention analysis."
        )
    )

    parser.add_argument("--woconf_report", required=True)
    parser.add_argument("--full_report", required=True)

    parser.add_argument("--woconf_pred_labels", required=True)
    parser.add_argument("--full_pred_labels", required=True)

    parser.add_argument("--woconf_gt_labels", required=True)
    parser.add_argument("--full_gt_labels", required=True)

    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of valid bootstrap resamples (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed (default: 2026).",
    )
    parser.add_argument(
        "--out_dir",
        default="output_bootstrap",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = prepare_data(args)

    print("========== Paired cluster bootstrap setup ==========")
    print(f"Evaluation rows: {len(data)}")
    print(f"Unique studies/clusters: {data['_cluster'].nunique()}")
    print(f"Requested valid bootstrap resamples: {args.n_boot}")
    print(f"Random seed: {args.seed}")

    # Point estimate on the full evaluation set.
    point = evaluate_sample(data)
    if point is None:
        raise RuntimeError(
            "Cannot construct decorrelation weights on the full dataset."
        )

    print("\n========== Full-sample point estimates ==========")
    print(
        f"w/o Conf: Pair-F1 Orig={point['wo_pair_f1_orig']:.6f}, "
        f"Pair-F1 Decor={point['wo_pair_f1_decor']:.6f}, "
        f"Delta-F1={point['wo_delta_f1']:.6f}, "
        f"Residual Corr Decor={point['wo_residual_corr_decor']:.6f}"
    )
    print(
        f"Full:     Pair-F1 Orig={point['fu_pair_f1_orig']:.6f}, "
        f"Pair-F1 Decor={point['fu_pair_f1_decor']:.6f}, "
        f"Delta-F1={point['fu_delta_f1']:.6f}, "
        f"Residual Corr Decor={point['fu_residual_corr_decor']:.6f}"
    )
    print(
        f"Paired Delta-F1 reduction (w/o - Full)="
        f"{point['paired_delta_f1_reduction']:.6f}"
    )
    print(
        f"Paired residual-corr reduction (w/o - Full)="
        f"{point['paired_residual_corr_reduction']:.6f}"
    )

    # Build cluster -> row-index lookup.
    cluster_to_indices = {
        k: np.asarray(v.index, dtype=int)
        for k, v in data.groupby("_cluster", sort=False)
    }
    clusters = np.asarray(list(cluster_to_indices.keys()), dtype=object)
    n_clusters = len(clusters)

    rng = np.random.default_rng(args.seed)

    boot_rows = []
    attempts = 0
    max_attempts = max(args.n_boot * 10, args.n_boot + 100)

    while len(boot_rows) < args.n_boot and attempts < max_attempts:
        attempts += 1

        sampled_clusters = rng.choice(
            clusters,
            size=n_clusters,
            replace=True,
        )

        # Keep every evaluation row belonging to a selected study.
        # If a study is sampled multiple times, its complete row cluster
        # is duplicated the same number of times.
        idx = np.concatenate(
            [cluster_to_indices[c] for c in sampled_clusters]
        )

        sample = data.iloc[idx].copy()

        stats = evaluate_sample(sample)
        if stats is None:
            # Extremely unlikely with this dataset, but skip if one of
            # the four C/E strata disappears in a bootstrap replicate.
            continue

        stats["bootstrap_id"] = len(boot_rows) + 1
        boot_rows.append(stats)

        if len(boot_rows) % 100 == 0:
            print(
                f"Completed {len(boot_rows)}/{args.n_boot} valid resamples"
            )

    if len(boot_rows) < args.n_boot:
        raise RuntimeError(
            f"Only {len(boot_rows)} valid bootstrap replicates were "
            f"obtained after {attempts} attempts."
        )

    boot = pd.DataFrame(boot_rows)

    # --------------------------------------------------------
    # Bootstrap summaries
    # --------------------------------------------------------
    summary_specs = [
        (
            "wo_delta_f1",
            point["wo_delta_f1"],
        ),
        (
            "fu_delta_f1",
            point["fu_delta_f1"],
        ),
        (
            "paired_delta_f1_reduction",
            point["paired_delta_f1_reduction"],
        ),
        (
            "wo_residual_corr_decor",
            point["wo_residual_corr_decor"],
        ),
        (
            "fu_residual_corr_decor",
            point["fu_residual_corr_decor"],
        ),
        (
            "paired_residual_corr_reduction",
            point["paired_residual_corr_reduction"],
        ),
        (
            "wo_pair_f1_orig",
            point["wo_pair_f1_orig"],
        ),
        (
            "wo_pair_f1_decor",
            point["wo_pair_f1_decor"],
        ),
        (
            "fu_pair_f1_orig",
            point["fu_pair_f1_orig"],
        ),
        (
            "fu_pair_f1_decor",
            point["fu_pair_f1_decor"],
        ),
    ]

    summary = pd.DataFrame(
        [
            summarize_metric(name, point_value, boot[name].to_numpy())
            for name, point_value in summary_specs
        ]
    )

    # Directional bootstrap support for the two paired differences.
    p_delta_positive = float(
        np.mean(boot["paired_delta_f1_reduction"] > 0)
    )
    p_corr_positive = float(
        np.mean(boot["paired_residual_corr_reduction"] > 0)
    )

    print("\n========== Bootstrap 95% percentile CIs ==========")

    important = [
        "wo_delta_f1",
        "fu_delta_f1",
        "paired_delta_f1_reduction",
        "wo_residual_corr_decor",
        "fu_residual_corr_decor",
        "paired_residual_corr_reduction",
    ]

    for metric in important:
        row = summary[summary["metric"] == metric].iloc[0]
        print(
            f"{metric}: "
            f"point={row['point_estimate']:.6f}, "
            f"95% CI=[{row['ci95_low']:.6f}, "
            f"{row['ci95_high']:.6f}]"
        )

    print("\n========== Paired bootstrap directional support ==========")
    print(
        "P(Delta-F1_w/o > Delta-F1_full) = "
        f"{p_delta_positive:.4f}"
    )
    print(
        "P(ResidualCorr_w/o > ResidualCorr_full) = "
        f"{p_corr_positive:.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    replicates_path = os.path.join(
        args.out_dir,
        "bootstrap_intervention_replicates.csv",
    )
    boot.to_csv(replicates_path, index=False)

    summary_path = os.path.join(
        args.out_dir,
        "bootstrap_intervention_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    txt_path = os.path.join(
        args.out_dir,
        "bootstrap_intervention_summary.txt",
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "Study-level paired cluster bootstrap for "
            "Cardiomegaly-Pleural Effusion intervention\n"
        )
        f.write(
            "============================================================\n\n"
        )
        f.write(f"Evaluation rows: {len(data)}\n")
        f.write(
            f"Unique studies/clusters: "
            f"{data['_cluster'].nunique()}\n"
        )
        f.write(f"Bootstrap resamples: {args.n_boot}\n")
        f.write(f"Random seed: {args.seed}\n\n")

        f.write("Full-sample point estimates\n")
        f.write("---------------------------\n")
        f.write(
            f"w/o Conf Delta-F1 = "
            f"{point['wo_delta_f1']:.8f}\n"
        )
        f.write(
            f"Full Delta-F1 = "
            f"{point['fu_delta_f1']:.8f}\n"
        )
        f.write(
            f"Paired Delta-F1 reduction = "
            f"{point['paired_delta_f1_reduction']:.8f}\n"
        )
        f.write(
            f"w/o Conf Residual Corr Decor = "
            f"{point['wo_residual_corr_decor']:.8f}\n"
        )
        f.write(
            f"Full Residual Corr Decor = "
            f"{point['fu_residual_corr_decor']:.8f}\n"
        )
        f.write(
            f"Paired residual-corr reduction = "
            f"{point['paired_residual_corr_reduction']:.8f}\n\n"
        )

        f.write("95% percentile confidence intervals\n")
        f.write("-----------------------------------\n")
        for metric in important:
            row = summary[summary["metric"] == metric].iloc[0]
            f.write(
                f"{metric}: point={row['point_estimate']:.8f}, "
                f"mean={row['bootstrap_mean']:.8f}, "
                f"95% CI=[{row['ci95_low']:.8f}, "
                f"{row['ci95_high']:.8f}]\n"
            )

        f.write("\nDirectional bootstrap support\n")
        f.write("-----------------------------\n")
        f.write(
            "P(Delta-F1_w/o > Delta-F1_full) = "
            f"{p_delta_positive:.6f}\n"
        )
        f.write(
            "P(ResidualCorr_w/o > ResidualCorr_full) = "
            f"{p_corr_positive:.6f}\n"
        )

    print("\nSaved:")
    print(replicates_path)
    print(summary_path)
    print(txt_path)


if __name__ == "__main__":
    main()
