import os
import re
import json
import argparse
import numpy as np
import pandas as pd


def extract_study_id(item):
    if isinstance(item, dict):
        if "study_id" in item:
            try:
                return int(item["study_id"])
            except Exception:
                pass

        if "id" in item:
            s = str(item["id"])
            m = re.search(r"(?:^|[^0-9])s?(\d+)$", s)
            if m:
                return int(m.group(1))

        for key in ["image_path", "path", "image", "images"]:
            if key not in item:
                continue
            value = item[key]
            values = value if isinstance(value, list) else [value]
            for v in values:
                if isinstance(v, dict):
                    for subkey in ["path", "image_path", "id"]:
                        if subkey in v:
                            s = str(v[subkey])
                            m = re.search(r"(?:/|\\)s(\d+)(?:/|\\|$)", s)
                            if m:
                                return int(m.group(1))
                else:
                    s = str(v)
                    m = re.search(r"(?:/|\\)s(\d+)(?:/|\\|$)", s)
                    if m:
                        return int(m.group(1))
    return None


def binarize_label(series, mapping):
    x = pd.to_numeric(series, errors="coerce")
    if mapping == "uones":
        return ((x == 1) | (x == -1)).astype(int)
    if mapping == "positive_only":
        return (x == 1).astype(int)
    raise ValueError(f"Unknown mapping: {mapping}")


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / np.sum(w))


def weighted_corr(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.nan
    w = w / w_sum

    mx = np.sum(w * x)
    my = np.sum(w * y)

    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def main():
    parser = argparse.ArgumentParser(
        description="Construct a Cardiomegaly-Pleural Effusion decorrelation intervention on MIMIC-CXR test."
    )
    parser.add_argument(
        "--chexpert_csv",
        default="pretrain/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv"
    )
    parser.add_argument(
        "--annotation",
        default="data/mimic_cxr/annotation.json"
    )
    parser.add_argument(
        "--out_dir",
        default="Corr/output"
    )
    parser.add_argument(
        "--mapping",
        choices=["uones", "positive_only"],
        default="uones"
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.annotation, "r", encoding="utf-8") as f:
        ann = json.load(f)

    if "test" not in ann:
        raise KeyError("annotation.json does not contain a 'test' split.")

    test_items = ann["test"]
    study_ids = []
    failed = 0
    for item in test_items:
        sid = extract_study_id(item)
        if sid is None:
            failed += 1
        else:
            study_ids.append(int(sid))

    test_study_ids = sorted(set(study_ids))

    print("========== Test split matching ==========")
    print(f"Annotation test items: {len(test_items)}")
    print(f"Extracted unique test study IDs: {len(test_study_ids)}")
    print(f"Items without extractable study_id: {failed}")

    if len(test_study_ids) == 0:
        raise RuntimeError("No test study IDs could be extracted.")

    df = pd.read_csv(args.chexpert_csv)
    required_cols = ["subject_id", "study_id", "Cardiomegaly", "Pleural Effusion"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["study_id"] = pd.to_numeric(df["study_id"], errors="coerce")
    df = df[df["study_id"].isin(test_study_ids)].copy()
    df = df.drop_duplicates(subset=["study_id"]).reset_index(drop=True)

    matched_ids = set(df["study_id"].dropna().astype(int).tolist())
    matching_ratio = len(matched_ids) / max(len(test_study_ids), 1)

    print(f"Matched CheXpert test studies: {len(matched_ids)}")
    print(f"Study-ID matching ratio: {matching_ratio:.4f}")

    df["C"] = binarize_label(df["Cardiomegaly"], args.mapping)
    df["E"] = binarize_label(df["Pleural Effusion"], args.mapping)
    df["stratum"] = "G" + df["C"].astype(str) + df["E"].astype(str)

    n = len(df)
    if n == 0:
        raise RuntimeError("No matched test studies remain after filtering.")

    p_c1 = float(df["C"].mean())
    p_e1 = float(df["E"].mean())
    p_c = {0: 1.0 - p_c1, 1: p_c1}
    p_e = {0: 1.0 - p_e1, 1: p_e1}

    joint_counts = df.groupby(["C", "E"], observed=False).size().to_dict()

    print("\n========== Original joint distribution ==========")
    summary_rows = []
    for c in [0, 1]:
        for e in [0, 1]:
            count = int(joint_counts.get((c, e), 0))
            p_joint = count / n
            p_indep = p_c[c] * p_e[e]
            print(
                f"G{c}{e}: count={count}, "
                f"P(C={c},E={e})={p_joint:.6f}, "
                f"P(C={c})P(E={e})={p_indep:.6f}"
            )
            summary_rows.append({
                "C": c,
                "E": e,
                "stratum": f"G{c}{e}",
                "count": count,
                "original_joint_prob": p_joint,
                "independent_target_prob": p_indep,
            })

    if any(r["count"] == 0 for r in summary_rows):
        raise RuntimeError("At least one disease stratum is empty.")

    rho_original = float(np.corrcoef(df["C"], df["E"])[0, 1])

    weight_lookup = {}
    for c in [0, 1]:
        for e in [0, 1]:
            p_joint = joint_counts[(c, e)] / n
            q_target = p_c[c] * p_e[e]
            weight_lookup[(c, e)] = q_target / p_joint

    df["intervention_weight_raw"] = [
        weight_lookup[(int(c), int(e))]
        for c, e in zip(df["C"], df["E"])
    ]
    df["intervention_weight"] = (
        df["intervention_weight_raw"]
        / df["intervention_weight_raw"].mean()
    )

    w = df["intervention_weight"].to_numpy(dtype=float)
    p_c1_decor = weighted_mean(df["C"], w)
    p_e1_decor = weighted_mean(df["E"], w)
    rho_decor = weighted_corr(df["C"], df["E"], w)

    weighted_total = float(np.sum(w))
    for row in summary_rows:
        mask = (df["C"] == row["C"]) & (df["E"] == row["E"])
        row["intervention_joint_prob"] = float(
            np.sum(w[mask.to_numpy()]) / weighted_total
        )
        row["importance_weight"] = weight_lookup[(row["C"], row["E"])]

    print("\n========== Intervention check ==========")
    print(
        "Cardiomegaly prevalence: "
        f"original={p_c1:.6f}, decorrelated={p_c1_decor:.6f}"
    )
    print(
        "Pleural Effusion prevalence: "
        f"original={p_e1:.6f}, decorrelated={p_e1_decor:.6f}"
    )
    print(
        "GT correlation: "
        f"original={rho_original:.6f}, decorrelated={rho_decor:.6f}"
    )

    print("\n========== Intervention weights ==========")
    for c in [0, 1]:
        for e in [0, 1]:
            print(f"G{c}{e}: weight={weight_lookup[(c, e)]:.6f}")

    manifest_cols = [
        "subject_id", "study_id", "Cardiomegaly", "Pleural Effusion",
        "C", "E", "stratum", "intervention_weight_raw", "intervention_weight"
    ]

    manifest_path = os.path.join(
        args.out_dir, "cardio_effusion_intervention_manifest.csv"
    )
    df[manifest_cols].to_csv(manifest_path, index=False)

    strata_path = os.path.join(
        args.out_dir, "cardio_effusion_intervention_strata.csv"
    )
    pd.DataFrame(summary_rows).to_csv(strata_path, index=False)

    summary_path = os.path.join(
        args.out_dir, "cardio_effusion_intervention_summary.txt"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Cardiomegaly-Pleural Effusion intervention summary\n")
        f.write("=================================================\n")
        f.write(f"mapping={args.mapping}\n")
        f.write(f"n_test_studies={n}\n")
        f.write(f"matching_ratio={matching_ratio:.6f}\n")
        f.write(f"cardiomegaly_prev_original={p_c1:.8f}\n")
        f.write(f"cardiomegaly_prev_decorrelated={p_c1_decor:.8f}\n")
        f.write(f"pleural_effusion_prev_original={p_e1:.8f}\n")
        f.write(f"pleural_effusion_prev_decorrelated={p_e1_decor:.8f}\n")
        f.write(f"gt_corr_original={rho_original:.8f}\n")
        f.write(f"gt_corr_decorrelated={rho_decor:.8f}\n")

    print("\nSaved:")
    print(manifest_path)
    print(strata_path)
    print(summary_path)


if __name__ == "__main__":
    main()
