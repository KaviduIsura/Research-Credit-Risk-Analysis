# Project brief for the coding agent

## Context
University coursework (BSc AI, KIC/NIBM, module "Machine Learning and Related Applications").
I reviewed the paper *Performance Evaluation of Machine Learning and Deep Learning Models for Credit Risk Prediction*.
- Phase 1 (done): `docs/ML_CW_Phase_1_report.docx`
- Brief: `docs/coursework_descriptor.pdf` and `docs/report_guide.docx`
- Phase 2 (this project, 25 marks): Option A, baseline implementation compared against the authors' results.
- Phase 3 (later): 2,500-word review report + this notebook as the technical appendix.

## Datasets (analysed separately)
- German Credit: 1,000 rows, 20 features (13 categorical), 30% bad credit.
- Taiwan Credit: 30,000 rows, 23 features (3 categorical), 22.1% default.
- Files live in `data/raw/` (see the README there). Never commit or modify raw data.

## What the paper did (as reviewed in Phase 1)
10 models x 3 resampling methods (SMOTE, ENN, SMOTE-ENN), stratified 10-fold CV, nested 5-fold tuning, metrics: accuracy, recall,
specificity, precision, F1. Authors' headline results: German MLP + SMOTE-ENN F1 = 0.937; Taiwan Random Forest + SMOTE-ENN F1 = 0.610.

## Flaws found in Phase 1 that this project must FIX (not just repeat)
1. Encoding not stated, SMOTE interpolating over encoded categories -> one-hot + SMOTE-NC.
2. Scaling not stated -> in-fold standardisation after signed-log of skewed amounts.
3. Missing values, duplicates, undocumented codes unaddressed -> documented cleaning with counts.
4. Accuracy/F1 only, fixed 0.5 threshold -> MCC, PR-AUC, Brier, German 5:1 cost matrix, threshold tuning in-fold.
5. Single 10-fold run, no significance tests -> repeated CV + corrected resampled t-test.
6. Best-of-many selection bias -> primary pipeline declared in advance (config `PRIMARY`).
7. No fairness audit, no calibration check -> Stage E of the notebook.

## Deliverable
`notebooks/Phase2_Credit_Risk_Improved.ipynb` (clean, commented, reproducible; runs top-to-bottom with Restart & Run All).

## Definition of done
- Notebook runs end to end with `QUICK_TEST = False` on both datasets, outputs saved, results in `results/`.
- All tables/figures used in the report are exported to `results/`.
- Conclusions section filled in using only numbers the notebook printed.
