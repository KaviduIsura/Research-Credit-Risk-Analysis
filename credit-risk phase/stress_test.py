"""
=======================================================================
 PHASE 3 – SECTION D: STRESS-TESTING  (All 3 Hypotheses)
 Paper: Mapfumo & Shongwe (2026), JRFM 19(3), 210

 Run from the "credit-risk phase" folder:
     python3 stress_test.py

 Estimated total runtime:  8 – 15 minutes
 Results printed to terminal → copy-paste into Word report.
=======================================================================
"""

import sys, time, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              accuracy_score, make_scorer)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent

# ── Column definitions (from paper Table 1) ──────────────────────────────────
GERMAN_COLUMNS = [
    "StatusOfCheckingAccount","DurationInMonths","CreditHistory",
    "LoanPurpose","CreditAmount","SavingsAccountBonds","EmploymentDuration",
    "InstallmentRatePercentage","PersonalStatusSex","OtherDebtorsGuarantors",
    "ResidenceDuration","PropertyType","AgeInYears","OtherInstallmentPlans",
    "HousingType","ExistingCreditsCount","JobType","Dependents",
    "Telephone","ForeignWorker","Target"
]
GERMAN_NUMERICAL = [
    "DurationInMonths","CreditAmount","InstallmentRatePercentage",
    "ResidenceDuration","AgeInYears","ExistingCreditsCount","Dependents"
]
GERMAN_CATEGORICAL = [c for c in GERMAN_COLUMNS
                      if c not in GERMAN_NUMERICAL and c != "Target"]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def banner(msg):
    print("\n" + "=" * 68)
    print(f"  {msg}")
    print("=" * 68)

def run_cv(X, y, label, n_folds=10):
    """10-fold stratified CV with RF + SMOTE-ENN (authors' best pipeline)."""
    pipe = ImbPipeline([
        ("smoteenn", SMOTEENN(random_state=SEED)),
        ("clf",      RandomForestClassifier(
                         n_estimators=100, random_state=SEED, n_jobs=-1))
    ])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scorers = {
        "acc":  make_scorer(accuracy_score),
        "rec":  make_scorer(recall_score,    zero_division=0),
        "pre":  make_scorer(precision_score, zero_division=0),
        "f1":   make_scorer(f1_score,        zero_division=0),
    }
    t0  = time.time()
    cvr = cross_validate(pipe, X, y, cv=skf, scoring=scorers, n_jobs=1)
    elapsed = time.time() - t0
    return dict(
        label=label,
        accuracy  = cvr["test_acc"].mean(),
        recall    = cvr["test_rec"].mean(),
        precision = cvr["test_pre"].mean(),
        f1        = cvr["test_f1"].mean(),
        time_s    = elapsed,
    )

def print_results(rows, baseline_label):
    """Print copy-paste-ready table."""
    w = 42
    print(f"\n{'Condition':<{w}} {'Accuracy':>9} {'Recall':>8} {'Precision':>10} {'F1-Score':>9}")
    print("-" * 82)
    for r in rows:
        tag = " <-- BASELINE" if r["label"] == baseline_label else ""
        print(f"{r['label']:<{w}} {r['accuracy']:>9.3f} {r['recall']:>8.3f} "
              f"{r['precision']:>10.3f} {r['f1']:>9.3f}{tag}")

def d(new, base):
    v = new - base
    return f"{'+'if v>=0 else ''}{v:.3f}"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
banner("LOADING GERMAN CREDIT DATASET")
data_path = BASE / "data" / "raw" / "german.data"
df = pd.read_csv(data_path, sep=r"\s+", names=GERMAN_COLUMNS, engine="python")
df["Target"] = df["Target"].map({1: 0, 2: 1})       # 0=Good, 1=Bad/Default
X_raw = df.drop(columns=["Target"])
y_ser = df["Target"].astype(int)

X_enc = X_raw.copy()
for c in GERMAN_CATEGORICAL:
    X_enc[c] = LabelEncoder().fit_transform(X_enc[c].astype(str))
X_base = X_enc.values.astype(float)
y_arr  = y_ser.values

print(f"  Dataset  : {X_base.shape[0]} instances, {X_base.shape[1]} features")
print(f"  Defaulters: {y_arr.sum()} / {len(y_arr)}  ({y_arr.mean()*100:.1f}%)")

# =============================================================================
#  HYPOTHESIS 1 — HIGH DIMENSIONALITY
# =============================================================================
banner("HYPOTHESIS 1 : High Dimensionality Stress-Test")
print("  Testing how RF+SMOTE-ENN performs as features grow via")
print("  degree-2 polynomial interaction expansion.")
h1 = []

print("\n  [1/2]  Baseline (20 original features) ...")
r = run_cv(X_base, y_arr, "Baseline   (20 features)")
h1.append(r)
print(f"         F1={r['f1']:.3f}  Recall={r['recall']:.3f}  [{r['time_s']:.1f}s]")

print("\n  [2/2]  Degree-2 polynomial expansion ...")
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_base)
poly    = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly  = poly.fit_transform(X_sc)
n_feat  = X_poly.shape[1]
print(f"         Expanded shape: {X_poly.shape}")
r = run_cv(X_poly, y_arr, f"High-Dim   ({n_feat} features, degree-2 poly)")
h1.append(r)
print(f"         F1={r['f1']:.3f}  Recall={r['recall']:.3f}  [{r['time_s']:.1f}s]")

print("\n" + "-"*50)
print("  >> COPY TABLE BELOW INTO REPORT SECTION D.1 <<")
print_results(h1, "Baseline   (20 features)")
print(f"\n  F1 change    : {d(h1[1]['f1'],    h1[0]['f1'])}  "
      f"({(h1[1]['f1']-h1[0]['f1'])/h1[0]['f1']*100:+.1f}% relative)")
print(f"  Recall change: {d(h1[1]['recall'], h1[0]['recall'])}")

# =============================================================================
#  HYPOTHESIS 2 — GAUSSIAN NOISE INJECTION
# =============================================================================
banner("HYPOTHESIS 2 : Gaussian Noise Injection Stress-Test")
print("  Adding zero-mean Gaussian noise at σ = k × per-feature std.")
print("  Levels: 0%, 10%, 20%, 30%, 50%\n")
h2 = []
noise_levels = [0.0, 0.10, 0.20, 0.30, 0.50]

for i, sf in enumerate(noise_levels):
    lbl = ("sigma=0%  (clean baseline)"
           if sf == 0 else f"sigma={int(sf*100)}% of feature std")
    print(f"  [{i+1}/{len(noise_levels)}]  {lbl} ...")
    if sf == 0:
        X_n = X_base.copy()
    else:
        noise = np.random.normal(0, sf * X_base.std(axis=0), X_base.shape)
        X_n   = X_base + noise
    r = run_cv(X_n, y_arr, lbl)
    h2.append(r)
    print(f"         F1={r['f1']:.3f}  Recall={r['recall']:.3f}  [{r['time_s']:.1f}s]")

print("\n" + "-"*50)
print("  >> COPY TABLE BELOW INTO REPORT SECTION D.2 <<")
print_results(h2, "sigma=0%  (clean baseline)")
print("\n  F1 degradation at each noise level:")
for r in h2[1:]:
    print(f"    {r['label']:<35}  ΔF1={d(r['f1'], h2[0]['f1'])}")

# =============================================================================
#  HYPOTHESIS 3 — EXTREME CLASS IMBALANCE
# =============================================================================
banner("HYPOTHESIS 3 : Extreme Class Imbalance Stress-Test")
print("  Progressively downsampling minority (default) class.\n")
h3 = []
ratios = [
    ("70:30  (original)", None),
    ("80:20",             0.20),
    ("90:10",             0.10),
    ("95:5",              0.05),
]

for i, (lbl, min_frac) in enumerate(ratios):
    print(f"  [{i+1}/{len(ratios)}]  Ratio {lbl} ...")
    if min_frac is None:
        X_u, y_u = X_base.copy(), y_arr.copy()
    else:
        idx_maj = np.where(y_arr == 0)[0]
        idx_min = np.where(y_arr == 1)[0]
        n_min_t = max(10, int(min_frac * len(idx_maj) / (1 - min_frac)))
        n_min_t = min(n_min_t, len(idx_min))
        rng     = np.random.RandomState(SEED)
        sel_min = rng.choice(idx_min, size=n_min_t, replace=False)
        all_idx = np.concatenate([idx_maj, sel_min])
        X_u, y_u = X_base[all_idx], y_arr[all_idx]
    actual = np.mean(y_u)*100
    print(f"         Shape: {X_u.shape}  |  Defaulters: {actual:.1f}%")
    r = run_cv(X_u, y_u, f"Ratio {lbl}")
    h3.append(r)
    print(f"         F1={r['f1']:.3f}  Recall={r['recall']:.3f}  [{r['time_s']:.1f}s]")

print("\n" + "-"*50)
print("  >> COPY TABLE BELOW INTO REPORT SECTION D.3 <<")
print_results(h3, "Ratio 70:30  (original)")
print("\n  Collapse at extreme imbalance:")
for r in h3[1:]:
    print(f"    {r['label']:<30}  ΔF1={d(r['f1'], h3[0]['f1'])}  "
          f"ΔRecall={d(r['recall'], h3[0]['recall'])}")

# =============================================================================
#  SUMMARY
# =============================================================================
total = sum(r["time_s"] for r in h1+h2+h3)
banner("FINAL SUMMARY  —  Copy numbers into your Word report")
print(f"\n  Total runtime: {total/60:.1f} minutes\n")
print(f"  H1 (Dimensionality) :")
print(f"     Baseline F1 (20 feat)   : {h1[0]['f1']:.3f}")
print(f"     High-Dim F1 ({n_feat} feat): {h1[1]['f1']:.3f}  (Δ {d(h1[1]['f1'],h1[0]['f1'])})")
print(f"\n  H2 (Noise) :")
print(f"     Clean F1  (sigma=0%) : {h2[0]['f1']:.3f}")
print(f"     Noisy F1  (sigma=50%): {h2[-1]['f1']:.3f}  (Δ {d(h2[-1]['f1'],h2[0]['f1'])})")
print(f"\n  H3 (Imbalance) :")
print(f"     70:30 F1: {h3[0]['f1']:.3f}  Recall: {h3[0]['recall']:.3f}")
print(f"     95:5  F1: {h3[-1]['f1']:.3f}  Recall: {h3[-1]['recall']:.3f}  "
      f"(Δ F1 {d(h3[-1]['f1'],h3[0]['f1'])})")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
out = BASE / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(h1).to_csv(out / "stress_test_h1_dimensionality.csv", index=False)
pd.DataFrame(h2).to_csv(out / "stress_test_h2_noise.csv",          index=False)
pd.DataFrame(h3).to_csv(out / "stress_test_h3_imbalance.csv",      index=False)
print(f"\n  CSVs saved → {out}")
print("\n✅  All 3 stress-tests complete!")
print("    Copy the tables above into Section D of your Word report.\n")
