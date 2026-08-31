import os
import argparse
import numpy as np
import pandas as pd

PAIR = ["Cardiomegaly", "Pleural Effusion"]


def to_binary_positive(series):
    """
    Match the positive-label convention used in the existing CE evaluator:
    only explicit positive labels are counted as 1; all other values are 0.
    Supports common numeric/string encodings.
    """
    s = series.copy()

    # Numeric path
    num = pd.to_numeric(s, errors="coerce")
    out = (num == 1).astype(int)

    # String path for non-numeric encodings
    text = s.astype(str).str.strip().str.lower()
    positive_tokens = {
        "positive", "pos", "present", "true", "yes", "1", "1.0"
    }
    out = ((out == 1) | text.isin(positive_tokens)).astype(int)
    return out


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


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / np.sum(w))


def pair_micro_prf(y_true, y_pred, w):
    """
    Weighted micro P/R/F1 over Cardiomegaly and Pleural Effusion.
    y_true/y_pred: [N, 2]
    w: [N] sample weights.
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
    return p, r, f1, tp, fp, fn


def read_report(path):
    df = pd.read_csv(path)
    need = ["subject_id", "study_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")

    df = df.copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="raise").astype("int64")
    df["study_id"] = pd.to_numeric(df["study_id"], errors="raise").astype("int64")
    return df


def attach_label_file(label_path, report_df, prefix):
    """
    Label CSV may:
    1) contain subject_id/study_id -> merge by IDs; or
    2) contain only disease columns -> attach by row order, requiring equal length.
    """
    lab = pd.read_csv(label_path)

    missing = [c for c in PAIR if c not in lab.columns]
    if missing:
        raise KeyError(f"{label_path}: missing disease columns {missing}")

    has_study = "study_id" in lab.columns

    if has_study:
        lab = lab.copy()
        lab["study_id"] = pd.to_numeric(lab["study_id"], errors="raise").astype("int64")

        if "subject_id" in lab.columns:
            lab["subject_id"] = pd.to_numeric(
                lab["subject_id"], errors="raise"
            ).astype("int64")
            keys = ["subject_id", "study_id"]
        else:
            keys = ["study_id"]

        use = lab[keys + PAIR].copy()
        merged = report_df[["subject_id", "study_id"]].merge(
            use, how="left", on=keys, validate="many_to_one"
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

    merged[f"{prefix}_C"] = to_binary_positive(merged["Cardiomegaly"])
    merged[f"{prefix}_E"] = to_binary_positive(merged["Pleural Effusion"])

    return merged[
        ["subject_id", "study_id", f"{prefix}_C", f"{prefix}_E"]
    ].copy()


def build_decorrelation_weights(gt):
    c = gt["gt_C"].to_numpy(dtype=int)
    e = gt["gt_E"].to_numpy(dtype=int)
    n = len(gt)

    p_c1 = float(np.mean(c))
    p_e1 = float(np.mean(e))
    p_c = {0: 1.0 - p_c1, 1: p_c1}
    p_e = {0: 1.0 - p_e1, 1: p_e1}

    joint = {}
    for cv in [0, 1]:
        for ev in [0, 1]:
            joint[(cv, ev)] = int(np.sum((c == cv) & (e == ev)))

    if any(v == 0 for v in joint.values()):
        raise RuntimeError(
            f"At least one C/E stratum is empty: {joint}. "
            "Cannot construct independence weights."
        )

    lookup = {}
    for cv in [0, 1]:
        for ev in [0, 1]:
            p_joint = joint[(cv, ev)] / n
            q_target = p_c[cv] * p_e[ev]
            lookup[(cv, ev)] = q_target / p_joint

    w = np.array(
        [lookup[(int(cv), int(ev))] for cv, ev in zip(c, e)],
        dtype=float,
    )
    w /= np.mean(w)

    info = {
        "n": n,
        "joint": joint,
        "weights": lookup,
        "c_prev_orig": p_c1,
        "e_prev_orig": p_e1,
        "c_prev_decor": weighted_mean(c, w),
        "e_prev_decor": weighted_mean(e, w),
        "gt_corr_orig": float(np.corrcoef(c, e)[0, 1]),
        "gt_corr_decor": weighted_corr(c, e, w),
    }
    return w, info


def evaluate_model(name, gt, pred, w_decor, info):
    y_true = gt[["gt_C", "gt_E"]].to_numpy(dtype=int)
    y_pred = pred[["pred_C", "pred_E"]].to_numpy(dtype=int)

    w_orig = np.ones(len(gt), dtype=float)

    p_o, r_o, f_o, tp_o, fp_o, fn_o = pair_micro_prf(
        y_true, y_pred, w_orig
    )
    p_d, r_d, f_d, tp_d, fp_d, fn_d = pair_micro_prf(
        y_true, y_pred, w_decor
    )

    pc = pred["pred_C"].to_numpy(dtype=int)
    pe = pred["pred_E"].to_numpy(dtype=int)

    pred_corr_orig = weighted_corr(pc, pe, w_orig)
    pred_corr_decor = weighted_corr(pc, pe, w_decor)

    cte_orig = abs(pred_corr_orig - info["gt_corr_orig"])
    cte_decor = abs(pred_corr_decor - info["gt_corr_decor"])

    return {
        "Model": name,
        "N": len(gt),
        "Pair_P_Orig": p_o,
        "Pair_R_Orig": r_o,
        "Pair_F1_Orig": f_o,
        "Pair_P_Decor": p_d,
        "Pair_R_Decor": r_d,
        "Pair_F1_Decor": f_d,
        "Delta_F1_abs": abs(f_d - f_o),
        "GT_Corr_Orig": info["gt_corr_orig"],
        "GT_Corr_Decor": info["gt_corr_decor"],
        "Pred_Corr_Orig": pred_corr_orig,
        "Pred_Corr_Decor": pred_corr_decor,
        "CTE_Orig": cte_orig,
        "CTE_Decor": cte_decor,
        "TP_Orig": tp_o,
        "FP_Orig": fp_o,
        "FN_Orig": fn_o,
        "TP_Decor": tp_d,
        "FP_Decor": fp_d,
        "FN_Decor": fn_d,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Intervention sensitivity analysis using the already-generated "
            "CE CheXbert prediction/GT label files."
        )
    )

    parser.add_argument("--woconf_report", required=True)
    parser.add_argument("--full_report", required=True)

    parser.add_argument("--woconf_pred_labels", required=True)
    parser.add_argument("--full_pred_labels", required=True)

    parser.add_argument("--woconf_gt_labels", required=True)
    parser.add_argument("--full_gt_labels", required=True)

    parser.add_argument("--out_dir", default="Corr/output_ce")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wo_report = read_report(args.woconf_report)
    fu_report = read_report(args.full_report)

    print("========== Report alignment ==========")
    print(
        f"w/o Conf: rows={len(wo_report)}, "
        f"unique studies={wo_report['study_id'].nunique()}"
    )
    print(
        f"Full:     rows={len(fu_report)}, "
        f"unique studies={fu_report['study_id'].nunique()}"
    )

    wo_keys = list(zip(wo_report["subject_id"], wo_report["study_id"]))
    fu_keys = list(zip(fu_report["subject_id"], fu_report["study_id"]))

    if wo_keys != fu_keys:
        raise RuntimeError(
            "The w/o-Conf and Full report.csv files do not contain the "
            "same (subject_id, study_id) sequence. Use outputs from the "
            "same MIMIC-CXR test split and order."
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
            "The two GT CheXbert files do not yield identical "
            "Cardiomegaly/Pleural Effusion labels."
        )

    gt = wo_gt

    wo_pred = attach_label_file(
        args.woconf_pred_labels, wo_report, prefix="pred"
    )
    fu_pred = attach_label_file(
        args.full_pred_labels, fu_report, prefix="pred"
    )

    w_decor, info = build_decorrelation_weights(gt)

    print("\n========== Evaluation-unit intervention ==========")
    print(f"N rows: {info['n']}")
    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(
            f"G{key[0]}{key[1]}: count={info['joint'][key]}, "
            f"weight={info['weights'][key]:.6f}"
        )

    print("\n========== Intervention check ==========")
    print(
        "Cardiomegaly prevalence: "
        f"original={info['c_prev_orig']:.6f}, "
        f"decorrelated={info['c_prev_decor']:.6f}"
    )
    print(
        "Pleural Effusion prevalence: "
        f"original={info['e_prev_orig']:.6f}, "
        f"decorrelated={info['e_prev_decor']:.6f}"
    )
    print(
        "GT correlation: "
        f"original={info['gt_corr_orig']:.6f}, "
        f"decorrelated={info['gt_corr_decor']:.6f}"
    )

    rows = [
        evaluate_model(
            "ECC+DyCE (w/o Conf)", gt, wo_pred, w_decor, info
        ),
        evaluate_model(
            "C3E-RRG (Full)", gt, fu_pred, w_decor, info
        ),
    ]
    res = pd.DataFrame(rows)

    show = [
        "Model",
        "Pair_P_Orig",
        "Pair_R_Orig",
        "Pair_F1_Orig",
        "Pair_F1_Decor",
        "Delta_F1_abs",
        "Pred_Corr_Orig",
        "Pred_Corr_Decor",
        "CTE_Decor",
    ]

    print("\n========== Intervention sensitivity results ==========")
    print(
        res[show].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        )
    )

    csv_path = os.path.join(
        args.out_dir, "cooccurrence_intervention_ce_results.csv"
    )
    res.to_csv(csv_path, index=False)

    txt_path = os.path.join(
        args.out_dir, "cooccurrence_intervention_ce_results.txt"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "Cardiomegaly-Pleural Effusion intervention sensitivity\n"
        )
        f.write(
            "=====================================================\n\n"
        )
        f.write(f"N={info['n']}\n")
        f.write(
            f"Cardiomegaly prevalence: "
            f"{info['c_prev_orig']:.8f} -> "
            f"{info['c_prev_decor']:.8f}\n"
        )
        f.write(
            f"Pleural Effusion prevalence: "
            f"{info['e_prev_orig']:.8f} -> "
            f"{info['e_prev_decor']:.8f}\n"
        )
        f.write(
            f"GT correlation: "
            f"{info['gt_corr_orig']:.8f} -> "
            f"{info['gt_corr_decor']:.8f}\n\n"
        )
        f.write(
            res[show].to_string(
                index=False, float_format=lambda x: f"{x:.6f}"
            )
        )
        f.write("\n")

    print("\nSaved:")
    print(csv_path)
    print(txt_path)


if __name__ == "__main__":
    main()
