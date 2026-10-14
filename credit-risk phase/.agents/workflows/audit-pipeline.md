---
description: Review the notebook for leakage and methodological mistakes
---
Review `notebooks/Phase2_Credit_Risk_Improved.ipynb` as a strict technical reviewer. Check, and report pass/fail with evidence for each:
1. Any preprocessing, scaling, resampling or threshold tuning performed outside the Pipeline / outside a training fold.
2. Same CV splits used across pipelines being compared (paired tests are valid).
3. Hyperparameter search and threshold tuning never see the outer validation fold.
4. Random seeds set everywhere; results cached with the configuration in the filename.
5. Metric definitions (positive class = bad credit / default; specificity, MCC, cost).
6. Anything in the markdown text that claims more than the code shows.
Do not change code; produce a short list of findings with line/cell references.
