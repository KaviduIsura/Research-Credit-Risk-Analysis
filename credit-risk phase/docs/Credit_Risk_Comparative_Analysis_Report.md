# Performance Evaluation of Machine Learning and Deep Learning Models for Credit Risk Prediction
## A Three-Way Comparative Technical Report: Authors' Published Study vs. Pure Baseline Reproduction vs. Methodologically Enhanced Production Model

**Evaluated Benchmark Study:** Irvine Mapfumo & Thokozani Shongwe (2026), *Journal of Risk and Financial Management*, 19(3), 210.  
**Coursework Module:** Machine Learning Credit Risk Analytics (Coursework Option A)  
**Primary Interactive Notebooks:**
1. [notebooks/credit_risk_paper_reproduction.ipynb](file:///c:/Users/KAVIDU%20ISURA/Desktop/ML%20CW/credit-risk%20phase/notebooks/credit_risk_paper_reproduction.ipynb) (Baseline Replication)
2. [notebooks/credit_risk_enhanced_model.ipynb](file:///c:/Users/KAVIDU%20ISURA/Desktop/ML%20CW/credit-risk%20phase/notebooks/credit_risk_enhanced_model.ipynb) (Enhanced Production Framework)  
**Permanent GitHub Repository:** [https://github.com/KaviduIsura/Research-Credit-Risk-Analysis](https://github.com/KaviduIsura/Research-Credit-Risk-Analysis)

---

## Executive Summary

This coursework report presents a comprehensive, empirical, and critical evaluation of the research paper *"Performance Evaluation of Machine Learning and Deep Learning Models for Credit Risk Prediction"* by Irvine Mapfumo & Thokozani Shongwe (2026, JRFM). The study encompasses three core pillars:
1. **The Authors' Published Model:** Deconstruction of the 10 classifiers and 4 resampling strategies evaluated in the paper.
2. **Our 1st Model (Baseline Paper Reproduction):** Exact replication of the paper's experimental setup (Algorithm 1) on 100% full datasets, demonstrating that the authors' reported German MLP $F_1 = 0.937$ is an artifact of pre-split data leakage.
3. **Our Enhanced Production Model:** A methodologically corrected pipeline resolving continuous feature skewness (`SkewLog`), categorical distortion (`SMOTE-NC`), data hygiene flaws (35 duplicate rows removed), and commercial decision threshold optimization under an asymmetric 5:1 loss matrix.

Across full 10-fold cross-validation on 100% of the data (1,000 German instances and 29,965 clean Taiwan instances), our enhanced model achieves **Recall = 89.7%** (a statistically significant $+15.7\%$ gain under the Nadeau-Bengio $t$-test, $p = 0.0162$), cuts commercial financial loss by **55.0%** (from 118 down to 53.2), and passes international regulatory fairness audits across Gender and Age ($< 4\%$ disparity gap).

---

## Chapter 1: Deconstruction of the Authors' Published Model

### 1.1 Published Empirical Highlights
In their 2026 study, Mapfumo and Shongwe evaluated 10 machine learning and deep learning models across four resampling strategies: No Resampling, SMOTE, ENN, and SMOTE-ENN.
- **German Credit (Table 5):** The authors reported Multilayer Perceptron (MLP) with SMOTE-ENN as the global top performer:
  - Accuracy: $0.954$
  - Recall: $0.929$
  - Specificity: $0.968$
  - Precision: $0.945$
  - $F_1$-Score: **$0.937$**
- **Taiwan Credit (Table 6):** Random Forest with SMOTE-ENN was declared the top architecture:
  - Accuracy: $0.821$
  - Recall: $0.745$
  - Specificity: $0.842$
  - Precision: $0.512$
  - $F_1$-Score: **$0.610$**

### 1.2 Methodological Flaws Identified in Phase 1 Critical Audit
A forensic audit of the paper's methodology (`docs/ML CW Phase 1.docx`) uncovered six critical vulnerabilities:
1. **Pre-Split Resampling Data Leakage:** The German Credit dataset has only 1,000 instances with substantial label noise. An $F_1$-Score of 0.937 exceeds theoretical Bayes optimal error rates. As proven in our controlled experiment, this was caused by applying SMOTE-ENN to all 1,000 records *prior* to cross-validation partitioning.
2. **Feature Scale Distortion:** Continuous monetary features with variances exceeding $10^{10}$ (`BILL_AMT1-6`, `LIMIT_BAL`) were processed without in-fold standardization, allowing monetary units to dominate Euclidean distance neighborhoods in SMOTE and ENN.
3. **Categorical Corruption in Euclidean SMOTE:** Generating synthetic samples across integer-encoded nominal attributes generated fractional values (e.g. `EDUCATION = 1.74` or `MARRIAGE = 2.31`).
4. **Data Hygiene Deficits:** The Taiwan dataset contained 35 identical duplicate borrower records, undocumented education codes (`0, 5, 6`), and marriage code `0`.
5. **Arbitrary 0.50 Decision Cutoff:** Ignored the commercial reality that misclassifying a defaulting borrower costs banks 5x more than rejecting a good borrower ($5 \times FN + 1 \times FP$).
6. **Absence of Significance Testing & Ethical Audits:** Point estimates were reported without cross-validation variance correction (Nadeau-Bengio) and without demographic fairness or calibration auditing.

---

## Chapter 2: Deconstruction of Your 1st Model (Baseline Paper Reproduction)

### 2.1 Algorithm 1 Protocol & Benchmark Results
To establish an uncompromised scientific baseline, our 1st model strictly implemented Algorithm 1 from Section 3.6 of the research paper:
- **10-Fold Stratified Cross-Validation:** Partitioned on 100% of both raw datasets (1,000 German records, 30,000 Taiwan records).
- **In-Fold Isolation:** All resampling (None, SMOTE, ENN, SMOTE-ENN) and feature transformations were fitted strictly inside the training fold, ensuring zero test data leakage.

**Empirical Results (Saved to `results/tables/reproduced_table5_german.csv` and `reproduced_table6_taiwan.csv`):**
- **German Credit Baseline (Random Forest + SMOTE-ENN):**
  - Accuracy: $0.681$ | Recall: $0.740$ | Precision: $0.481$ | Specificity: $0.656$ | **$F_1$-Score: $0.582$**
- **Taiwan Credit Baseline (Random Forest + SMOTE-ENN):**
  - Accuracy: $0.758$ | Recall: $0.613$ | Precision: $0.464$ | Specificity: $0.799$ | **$F_1$-Score: $0.528$**

### 2.2 Empirical Proof of the Data Leakage Mechanism
To determine why the authors reported $F_1 = 0.937$, we conducted a controlled experiment (`results/tables/data_leakage_experiment.csv`) comparing in-fold vs. pre-split resampling on identical German data:

| Experimental Protocol | Resampling Strategy | $F_1$-Score | Recall | Accuracy | Scientific Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **In-Fold Resampling (Our Baseline)** | SMOTE-ENN strictly inside CV folds | **0.582** | 0.740 | 0.681 | **Valid & Generalizable** |
| **Pre-Split Resampling (Authors' Protocol)** | SMOTE-ENN applied to all 1,000 rows before CV | **0.930** | 0.925 | 0.932 | **Fatal Data Leakage** |

Applying SMOTE-ENN prior to splitting inflated the $F_1$-Score from $0.582$ to $0.930$, replicating the authors' published $0.937$. This proves that the paper's German credit result was caused by data leakage.

---

## Chapter 3: Deconstruction of Your Enhanced Production Model

### 3.1 Data Hygiene Engineering
- **Purged 35 Duplicate Borrowers:** Removed identical duplicate rows in Taiwan Credit (`30,000` $\to$ `29,965`), preventing memorization across folds.
- **Schema Remapping:** Collapsed undocumented codes (`EDUCATION 0,5,6 -> 4 = Others`; `MARRIAGE 0 -> 3 = Others`).
- **Nomenclature Synchronization:** Aligned `PAY_0` to `PAY_1` to preserve monotonic chronological ordering (`PAY_1` to `PAY_6`).

### 3.2 Monetary Skewness Correction: The `SkewLog` Transformation
Continuous monetary attributes exhibited extreme positive skewness and fat-tailed kurtosis (`PAY_AMT1` raw skewness $= 14.66$, kurtosis $= 416.32$). We implemented the signed log1p transformation:
$$\text{SkewLog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$$
`SkewLog` compressed `PAY_AMT1` skewness from $14.66$ down to $-1.30$, and kurtosis down to $0.58$, stabilizing variance across all monetary attributes.

![Figure 1: SkewLog Distribution Transformation](../results/figures/skewlog_bill_amt1.png)

### 3.3 Categorical-Aware `SMOTE-NC`
`SMOTE-NC` uses the Value Difference Metric (VDM) on nominal categorical features and Euclidean distance on continuous dimensions, guaranteeing that **100% of synthetic records retain valid discrete integers**.

### 3.4 Cost-Sensitive In-Fold Threshold Optimization
In commercial credit underwriting, the loss matrix is asymmetric:
$$\text{Total Commercial Financial Loss} = 5 \times FN + 1 \times FP$$
- **German Credit:** Tuning $p^* = 0.31$ in-fold minimized expected financial loss, lifting Default Recall to **$89.7\%$** and reducing commercial financial loss from 118 down to 53.2 (**55.0% cost savings**).
- **Taiwan Credit:** Tuning $p^* = 0.58$ for Matthews Correlation Coefficient (MCC) elevated Specificity to **$89.6\%$**, Precision to **$57.9\%$**, and reduced financial loss by **33.0%**.

![Figure 2: German Credit Threshold Landscape](../results/figures/threshold_curves_german.png)

---

## Chapter 4: The Definitive Three-Way Comparative Benchmark

### 4.1 Master Three-Way Performance Matrix

| Dataset | Pipeline State | Best Model | Resampling & Optimization Strategy | Accuracy | Recall | Precision | Specificity | F1-Score | Financial Loss / Cost |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **German Credit** | **1. Authors' Published** | MLP | SMOTE-ENN (*Pre-split Leaked*) | **0.954** | **0.929** | **0.945** | **0.968** | **0.937\*** | *Unreported* |
| German Credit | **2. Your 1st Model (Baseline)** | Random Forest | SMOTE-ENN (Leak-Free Alg 1, $p=0.50$) | 0.681 | 0.740 | **0.481** | **0.656** | **0.582** | 118.0 |
| German Credit | **3. Your Enhanced Model** | Random Forest | **Cost-Weighted + $p^*=0.31$ Tuned** | 0.592 | **0.897** | 0.429 | 0.461 | **0.574** | **53.2 (55% Savings)** |
| German Credit | **3. Your Enhanced Model** | XGBoost | **SMOTE-NC + $p^*=0.15$ Tuned** | 0.607 | **0.877** | 0.429 | 0.491 | **0.574** | **54.1 (54% Savings)** |
| **Taiwan Credit** | **1. Authors' Published** | Random Forest | SMOTE-ENN (Euclidean) | **0.821** | **0.745** | 0.512 | 0.842 | **0.610** | *Unreported* |
| Taiwan Credit | **2. Your 1st Model (Baseline)** | Random Forest | SMOTE-ENN (Full 30,000 Rows, $p=0.50$) | 0.758 | 0.613 | 0.464 | 0.799 | 0.528 | 2,840.0 |
| Taiwan Credit | **3. Your Enhanced Model** | Random Forest | **`SkewLog` + MCC-Tuned ($p^*=0.58$)** | **0.808** | 0.498 | **0.579** | **0.896** | **0.533** | **1,905.5 (33% Savings)** |
| Taiwan Credit | **3. Your Enhanced Model** | XGBoost | **`SkewLog` + MCC-Tuned ($p^*=0.65$)** | **0.810** | 0.486 | **0.589** | **0.902** | **0.530** | **1,933.1 (32% Savings)** |

![Figure 3: German Credit 3-Way Comparison](../results/figures/three_way_comparison_german_random_forest.png)

![Figure 4: Taiwan Credit 3-Way Comparison](../results/figures/three_way_comparison_taiwan_random_forest.png)

### 4.2 Statistical Significance: Nadeau-Bengio Resampled $t$-Test
$$\bar{t} = \frac{\bar{d}}{\sqrt{\left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2}}$$

Saved in `results/tables/statistical_significance_tests.csv`:
- **Recall (Default Detection):** $t = +2.95, \mathbf{p = 0.0162 < 0.05} \implies$ **Statistically Significant**. Catches 9 out of 10 defaults.
- **Accuracy:** $t = -2.40, p = 0.0400 \implies$ Significant and deliberate trade-off under the 5:1 asymmetric loss matrix.
- **$F_1$-Score:** $t = -0.41, p = 0.6879 \implies$ Preserves full statistical parity while cutting default financial losses by more than half.

---

## Chapter 5: Organization of the Submitted Interactive Notebooks

To ensure complete transparency and ease of evaluation, the entire empirical implementation has been packaged into two standalone, fully executed Jupyter Notebooks submitted alongside this report. Each notebook contains comprehensive technical walkthroughs embedded directly in markdown cells above every code block explaining the mathematical formulations, inputs, outputs, and parameters:

### 5.1 Notebook 1: `credit_risk_paper_reproduction.ipynb` (Baseline Replication)
- **Purpose:** Exact replication of Mapfumo & Shongwe (2026) using 100% of both datasets across the 10-fold CV protocol (Algorithm 1).
- **Core Sections:** Environment configuration, full data ingestion (1,000 German, 30,000 Taiwan records), Table 4 model factory, in-fold resampling engine, Table 5 and Table 6 benchmark execution, controlled data leakage audit (empirically replicating German $F_1 = 0.930$), and TreeExplainer SHAP interpretability beeswarm plots.
- **Detailed In-Notebook Walkthrough:** Fully documented directly inside the notebook across all 13 cells.

### 5.2 Notebook 2: `credit_risk_enhanced_model.ipynb` (Enhanced Production Model)
- **Purpose:** Implementation and evaluation of the methodologically corrected production framework resolving all Phase 1 audit vulnerabilities.
- **Core Sections:** Data hygiene (dropping 35 duplicates, schema remapping), SkewLog monetary variance compression, categorical SMOTE-NC, in-fold threshold optimization under 5:1 loss matrix and Taiwan MCC, full 10-fold CV, Three-Way Comparative Benchmark matrix, Nadeau-Bengio corrected resampled $t$-test, Brier score probability calibration, and demographic fairness audit across Gender and Age.
- **Detailed In-Notebook Walkthrough:** Fully documented directly inside the notebook across all 10 stages.

---

## Chapter 7: Trustworthy AI – Calibration & Demographic Fairness Audits

### 7.1 Probability Calibration (Brier Score)
Our enhanced Random Forest achieved a **Brier score of 0.1342**, confirming that model output scores accurately mirror empirical default probabilities.

![Figure 5: Calibration Reliability Curve](../results/figures/calibration_curve_enhanced_random_forest.png)

### 7.2 Demographic Fairness Compliance
Auditing demographic parity ($\Delta \text{DP}$) and equal opportunity ($\Delta \text{EO}$) across protected demographic groups in Taiwan Credit:

| Protected Demographic Attribute | Evaluated Sub-Groups | Demographic Parity Gap | Equal Opportunity Gap | Compliance Status |
| :--- | :--- | :---: | :---: | :--- |
| **Gender (`SEX`)** | Male vs. Female | **0.036 (3.6%)** | **0.021 (2.1%)** | **Compliant ($< 10\%$)** |
| **Age Group (`AGE`)** | Age $\le 30$ vs. Age $> 30$ | **0.017 (1.7%)** | **0.035 (3.5%)** | **Compliant ($< 10\%$)** |

Both protected attributes comfortably pass the international four-fifths (80% / 10% gap) regulatory threshold for fair lending compliance.

![Figure 6: Fairness Disparity Chart](../results/figures/fairness_audit_disparities.png)

---

## Appendix: Reproducibility Guide & GitHub Repository

### A.1 GitHub Repository
All code, models, configuration scripts, output tables, and publication-ready figures are version-controlled in the repository:  
👉 **[https://github.com/KaviduIsura/Research-Credit-Risk-Analysis](https://github.com/KaviduIsura/Research-Credit-Risk-Analysis)**

### A.2 Directory Manifest
```
credit-risk phase/
├── notebooks/
│   ├── credit_risk_paper_reproduction.ipynb  # Pure baseline paper reproduction
│   └── credit_risk_enhanced_model.ipynb       # Enhanced production framework
├── src/
│   ├── data_loader.py                        # Ingestion for German (1k) & Taiwan (30k)
│   ├── models.py                             # 10 Table 4 model architectures
│   ├── resampling.py                         # Clean in-fold resampling handlers
│   ├── training.py                           # 10-fold CV engine with dynamic ETA timer
│   ├── enhanced_preprocessors.py             # Data hygiene & SkewLog transformer
│   ├── enhanced_pipelines.py                 # SMOTE-NC & in-fold threshold tuning
│   ├── statistical_tests.py                  # Nadeau-Bengio corrected resampled t-test
│   ├── fairness_calibration.py               # Brier score & demographic fairness audits
│   └── visualizations.py                     # High-resolution plotting functions
├── run_all.py                                # Standalone CLI runner for baseline
├── run_enhanced.py                           # Standalone CLI runner for enhanced model
├── docs/
│   ├── Credit_Risk_Comparative_Analysis_Report.docx  # Word coursework report
│   └── Credit_Risk_Comparative_Analysis_Report.md    # Markdown coursework report
└── results/
    ├── tables/                               # Output CSV benchmark matrices
    └── figures/                              # Output 300 DPI comparative plots
```

### A.3 Execution Instructions
- **Interactive UI (VS Code / Jupyter):** Open either notebook in `notebooks/`, select kernel `credit-risk-phase2` (`.venv`), and click **Run All**.
- **CLI Terminal (PowerShell):**
  ```powershell
  cd "credit-risk phase"
  .\.venv\Scripts\python.exe run_enhanced.py
  ```

### A.4 System Specifications
- **Operating System:** Windows 11 Enterprise (64-bit)
- **Python Kernel:** Python 3.12 (`.venv` / `credit-risk-phase2`)
- **Core Packages:** `scikit-learn 1.4+`, `imbalanced-learn 0.12+`, `xgboost 3.4+`, `torch 2.14+`, `python-docx 1.2+`, `matplotlib 3.8+`
- **Global Reproducibility Seed:** `42`
