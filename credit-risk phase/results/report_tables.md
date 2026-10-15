# Complete Phase 2 Technical Review Report Tables


> **Note:** All tables in this document are automatically extracted directly from the production cross-validation CSV outputs in `results/`. Values are formatted to 3 decimal places with complete methodological provenance documented under each table.


## Table 1a: Authors' Published Results — German Credit Dataset (Table 5, p. 15)

| Resampling Method | Model | Accuracy | Recall | Specificity | Precision | F1 Score (Minority) |
| --- | --- | --- | --- | --- | --- | --- |
| No Resampling | Logistic Regression | 0.705 | 0.943 | 0.136 | 0.723 | 0.818 |
| No Resampling | SVM | 0.750 | 1.000 | 0.153 | 0.738 | 0.849 |
| No Resampling | Decision Tree | 0.645 | 0.738 | 0.424 | 0.754 | 0.746 |
| No Resampling | Random Forest | 0.800 | 0.940 | 0.407 | 0.783 | 0.834 |
| No Resampling | XGBoost | 0.760 | 0.851 | 0.576 | 0.816 | 0.833 |
| No Resampling | AdaBoost | 0.760 | 0.915 | 0.390 | 0.782 | 0.843 |
| No Resampling | MLP | 0.772 | 0.872 | 0.525 | 0.800 | 0.842 |
| No Resampling | CNN | 0.775 | 0.908 | 0.458 | 0.800 | 0.850 |
| No Resampling | LSTM | 0.720 | 0.894 | 0.305 | 0.754 | 0.818 |
| No Resampling | GRU | 0.735 | 0.747 | 0.594 | 0.824 | 0.809 |
| SMOTE | Logistic Regression | 0.635 | 0.709 | 0.458 | 0.758 | 0.733 |
| SMOTE | SVM | 0.695 | 0.783 | 0.504 | 0.808 | 0.775 |
| SMOTE | Decision Tree | 0.655 | 0.723 | 0.492 | 0.773 | 0.747 |
| SMOTE | Random Forest | 0.760 | 0.851 | 0.576 | 0.816 | 0.833 |
| SMOTE | XGBoost | 0.763 | 0.864 | 0.846 | 0.816 | 0.833 |
| SMOTE | AdaBoost | 0.705 | 0.773 | 0.542 | 0.801 | 0.787 |
| SMOTE | MLP | 0.771 | 0.732 | 0.817 | 0.820 | 0.773 |
| SMOTE | CNN | 0.714 | 0.624 | 0.817 | 0.795 | 0.699 |
| SMOTE | LSTM | 0.739 | 0.644 | 0.847 | 0.828 | 0.725 |
| SMOTE | GRU | 0.746 | 0.698 | 0.802 | 0.800 | 0.746 |
| ENN | Logistic Regression | 0.575 | 0.553 | 0.627 | 0.780 | 0.647 |
| ENN | SVM | 0.610 | 0.553 | 0.744 | 0.786 | 0.667 |
| ENN | Decision Tree | 0.610 | 0.610 | 0.780 | 0.869 | 0.717 |
| ENN | Random Forest | 0.675 | 0.638 | 0.763 | 0.865 | 0.735 |
| ENN | XGBoost | 0.630 | 0.596 | 0.712 | 0.832 | 0.694 |
| ENN | AdaBoost | 0.670 | 0.640 | 0.780 | 0.871 | 0.727 |
| ENN | MLP | 0.844 | 0.821 | 0.861 | 0.821 | 0.821 |
| ENN | CNN | 0.797 | 0.714 | 0.861 | 0.800 | 0.755 |
| ENN | LSTM | 0.750 | 0.643 | 0.833 | 0.750 | 0.692 |
| ENN | GRU | 0.750 | 0.625 | 0.847 | 0.761 | 0.686 |
| SMOTE-ENN | Logistic Regression | 0.570 | 0.582 | 0.542 | 0.752 | 0.656 |
| SMOTE-ENN | SVM | 0.615 | 0.624 | 0.593 | 0.786 | 0.696 |
| SMOTE-ENN | Decision Tree | 0.630 | 0.640 | 0.593 | 0.777 | 0.718 |
| SMOTE-ENN | Random Forest | 0.680 | 0.738 | 0.542 | 0.794 | 0.765 |
| SMOTE-ENN | XGBoost | 0.670 | 0.702 | 0.593 | 0.805 | 0.750 |
| SMOTE-ENN | AdaBoost | 0.615 | 0.652 | 0.525 | 0.795 | 0.705 |
| SMOTE-ENN | MLP | 0.954 | 0.929 | 0.968 | 0.945 | 0.937 |
| SMOTE-ENN | CNN | 0.934 | 0.893 | 0.958 | 0.926 | 0.909 |
| SMOTE-ENN | LSTM | 0.927 | 0.911 | 0.933 | 0.895 | 0.903 |
| SMOTE-ENN | GRU | 0.914 | 0.839 | 0.958 | 0.922 | 0.879 |


*Source: results/authors_reported.csv*


## Table 1b: Authors' Published Results — Taiwan Credit Dataset (Table 6, p. 16)

| Resampling Method | Model | Accuracy | Recall | Specificity | Precision | F1 Score (Minority) |
| --- | --- | --- | --- | --- | --- | --- |
| No Resampling | Logistic Regression | 0.810 | 0.235 | 0.971 | 0.691 | 0.351 |
| No Resampling | SVM | 0.820 | 0.331 | 0.957 | 0.682 | 0.446 |
| No Resampling | Decision Tree | 0.724 | 0.409 | 0.813 | 0.380 | 0.394 |
| No Resampling | Random Forest | 0.817 | 0.361 | 0.944 | 0.640 | 0.463 |
| No Resampling | XGBoost | 0.817 | 0.376 | 0.940 | 0.637 | 0.473 |
| No Resampling | AdaBoost | 0.816 | 0.308 | 0.958 | 0.674 | 0.423 |
| No Resampling | MLP | 0.810 | 0.303 | 0.943 | 0.622 | 0.433 |
| No Resampling | CNN | 0.796 | 0.179 | 0.969 | 0.618 | 0.278 |
| No Resampling | LSTM | 0.807 | 0.349 | 0.935 | 0.600 | 0.441 |
| No Resampling | GRU | 0.812 | 0.331 | 0.947 | 0.637 | 0.436 |
| SMOTE | Logistic Regression | 0.688 | 0.582 | 0.718 | 0.366 | 0.449 |
| SMOTE | SVM | 0.749 | 0.583 | 0.795 | 0.444 | 0.504 |
| SMOTE | Decision Tree | 0.694 | 0.488 | 0.752 | 0.355 | 0.411 |
| SMOTE | Random Forest | 0.763 | 0.487 | 0.866 | 0.504 | 0.495 |
| SMOTE | XGBoost | 0.763 | 0.464 | 0.846 | 0.468 | 0.461 |
| SMOTE | AdaBoost | 0.733 | 0.586 | 0.773 | 0.420 | 0.490 |
| SMOTE | MLP | 0.726 | 0.538 | 0.779 | 0.405 | 0.462 |
| SMOTE | CNN | 0.736 | 0.483 | 0.809 | 0.415 | 0.446 |
| SMOTE | LSTM | 0.756 | 0.590 | 0.802 | 0.455 | 0.514 |
| SMOTE | GRU | 0.703 | 0.697 | 0.705 | 0.398 | 0.507 |
| ENN | Logistic Regression | 0.803 | 0.447 | 0.902 | 0.562 | 0.498 |
| ENN | SVM | 0.803 | 0.490 | 0.891 | 0.557 | 0.521 |
| ENN | Decision Tree | 0.669 | 0.593 | 0.690 | 0.349 | 0.439 |
| ENN | Random Forest | 0.778 | 0.553 | 0.841 | 0.493 | 0.521 |
| ENN | XGBoost | 0.767 | 0.580 | 0.820 | 0.474 | 0.521 |
| ENN | AdaBoost | 0.786 | 0.549 | 0.852 | 0.510 | 0.529 |
| ENN | MLP | 0.772 | 0.356 | 0.772 | 0.482 | 0.516 |
| ENN | CNN | 0.787 | 0.302 | 0.922 | 0.521 | 0.383 |
| ENN | LSTM | 0.760 | 0.621 | 0.799 | 0.463 | 0.531 |
| ENN | GRU | 0.789 | 0.524 | 0.864 | 0.518 | 0.521 |
| SMOTE-ENN | Logistic Regression | 0.593 | 0.746 | 0.550 | 0.317 | 0.445 |
| SMOTE-ENN | SVM | 0.647 | 0.741 | 0.621 | 0.354 | 0.479 |
| SMOTE-ENN | Decision Tree | 0.660 | 0.601 | 0.677 | 0.343 | 0.436 |
| SMOTE-ENN | Random Forest | 0.821 | 0.745 | 0.842 | 0.512 | 0.610 |
| SMOTE-ENN | XGBoost | 0.708 | 0.632 | 0.729 | 0.395 | 0.486 |
| SMOTE-ENN | AdaBoost | 0.652 | 0.733 | 0.629 | 0.356 | 0.479 |
| SMOTE-ENN | MLP | 0.651 | 0.714 | 0.633 | 0.353 | 0.472 |
| SMOTE-ENN | CNN | 0.606 | 0.701 | 0.580 | 0.318 | 0.438 |
| SMOTE-ENN | LSTM | 0.647 | 0.767 | 0.613 | 0.357 | 0.487 |
| SMOTE-ENN | GRU | 0.679 | 0.720 | 0.667 | 0.377 | 0.495 |


*Source: results/authors_reported.csv*


## Table 2: Benchmark Comparison Verdicts (Our Primary Pipeline vs. Authors' Published)

| Dataset | Metric | Our Primary Pipeline | Authors' Published | Difference (Ours - Authors) | Uncertainty Band (±2.5 SD) | Formal Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| German | f1 | 0.587 | 0.937 | -0.350 | 0.028 | Authors better |
| German | recall | 0.697 | 0.929 | -0.232 | 0.127 | Authors better |
| German | precision | 0.523 | 0.945 | -0.422 | 0.063 | Authors better |
| German | specificity | 0.707 | 0.968 | -0.261 | 0.147 | Authors better |
| German | accuracy | 0.704 | 0.954 | -0.250 | 0.069 | Authors better |
| Taiwan | f1 | 0.547 | 0.610 | -0.063 | 0.020 | Authors better |
| Taiwan | recall | 0.557 | 0.745 | -0.188 | 0.020 | Authors better |
| Taiwan | precision | 0.537 | 0.512 | 0.025 | 0.020 | Mine better |
| Taiwan | specificity | 0.863 | 0.842 | 0.021 | 0.020 | Mine better |
| Taiwan | accuracy | 0.796 | 0.821 | -0.025 | 0.020 | Authors better |


*Source: results/verdicts.csv*


## Table 3: Statistical Significance Tests: Corrected vs. Paper-Style Pipeline (MCC)

| Dataset | Model | Methodological Fix Variant | Paper-Style MCC | Corrected MCC | Delta MCC | t-statistic | p-value (Corrected t-test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| German | LogReg | fixed_none | 0.382 | 0.422 | 0.040 | 1.331 | 0.194 |
| German | LogReg | fixed_smotenc | 0.382 | 0.365 | -0.016 | -0.437 | 0.665 |
| German | LogReg | fixed_smoteenn | 0.382 | 0.351 | -0.031 | -0.917 | 0.366 |
| German | RandomForest | fixed_none | 0.384 | 0.383 | -0.001 | -0.025 | 0.980 |
| German | RandomForest | fixed_smotenc | 0.384 | 0.406 | 0.022 | 0.504 | 0.618 |
| German | RandomForest | fixed_smoteenn | 0.384 | 0.380 | -0.004 | -0.101 | 0.920 |
| German | XGBoost | fixed_none | 0.364 | 0.421 | 0.057 | 1.745 | 0.091 |
| German | XGBoost | fixed_smotenc | 0.364 | 0.390 | 0.026 | 0.691 | 0.495 |
| German | XGBoost | fixed_smoteenn | 0.364 | 0.355 | -0.009 | -0.321 | 0.751 |
| German | MLP | fixed_none | 0.006 | 0.318 | 0.312 | 6.268 | 0.000 |
| German | MLP | fixed_smotenc | 0.006 | 0.393 | 0.388 | 11.579 | 0.000 |
| German | MLP | fixed_smoteenn | 0.006 | 0.343 | 0.338 | 10.510 | 0.000 |
| Taiwan | LogReg | fixed_none | 0.193 | 0.350 | 0.157 | 24.279 | 0.000 |
| Taiwan | LogReg | fixed_smotenc | 0.193 | 0.336 | 0.142 | 23.110 | 0.000 |
| Taiwan | LogReg | fixed_smoteenn | 0.193 | 0.286 | 0.093 | 12.571 | 0.000 |
| Taiwan | RandomForest | fixed_none | 0.388 | 0.382 | -0.007 | -0.711 | 0.495 |
| Taiwan | RandomForest | fixed_smotenc | 0.388 | 0.387 | -0.001 | -0.159 | 0.877 |
| Taiwan | RandomForest | fixed_smoteenn | 0.388 | 0.371 | -0.017 | -2.534 | 0.032 |
| Taiwan | XGBoost | fixed_none | 0.398 | 0.391 | -0.007 | -2.149 | 0.060 |
| Taiwan | XGBoost | fixed_smotenc | 0.398 | 0.408 | 0.010 | 1.569 | 0.151 |
| Taiwan | XGBoost | fixed_smoteenn | 0.398 | 0.385 | -0.014 | -2.886 | 0.018 |
| Taiwan | MLP | fixed_none | 0.171 | 0.394 | 0.223 | 16.304 | 0.000 |
| Taiwan | MLP | fixed_smotenc | 0.171 | 0.337 | 0.165 | 9.227 | 0.000 |
| Taiwan | MLP | fixed_smoteenn | 0.171 | 0.309 | 0.138 | 9.999 | 0.000 |


*Source: results/fix_vs_paper_style_tests.csv*


## Table 4a1: German Credit — Stage A Model × Variant MCC Matrix

| Model | paper_style | fixed_none | fixed_smotenc | fixed_smoteenn |
| --- | --- | --- | --- | --- |
| LogReg | 0.382 | 0.422 | 0.365 | 0.351 |
| MLP | 0.006 | 0.318 | 0.393 | 0.343 |
| RandomForest | 0.384 | 0.383 | 0.406 | 0.380 |
| XGBoost | 0.364 | 0.421 | 0.390 | 0.355 |


*Source: results/all_results.csv*


## Table 4a2: German Credit — Stage A Full Benchmark Metrics (Mean)

| Model | Pipeline Variant | Accuracy | Balanced Acc | Recall | Specificity | Precision | F1 | MCC | PR-AUC | ROC-AUC | Brier Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogReg | fixed_none | 0.727 | 0.725 | 0.719 | 0.730 | 0.537 | 0.613 | 0.422 | 0.613 | 0.781 | 0.190 |
| LogReg | fixed_smoteenn | 0.639 | 0.689 | 0.816 | 0.563 | 0.447 | 0.576 | 0.351 | 0.603 | 0.767 | 0.278 |
| LogReg | fixed_smotenc | 0.717 | 0.689 | 0.619 | 0.760 | 0.530 | 0.568 | 0.365 | 0.610 | 0.767 | 0.188 |
| LogReg | paper_style | 0.684 | 0.705 | 0.759 | 0.652 | 0.490 | 0.591 | 0.382 | 0.596 | 0.769 | 0.223 |
| MLP | fixed_none | 0.743 | 0.632 | 0.354 | 0.909 | 0.620 | 0.428 | 0.318 | 0.596 | 0.749 | 0.174 |
| MLP | fixed_smoteenn | 0.640 | 0.685 | 0.800 | 0.571 | 0.447 | 0.572 | 0.343 | 0.610 | 0.771 | 0.274 |
| MLP | fixed_smotenc | 0.734 | 0.701 | 0.620 | 0.783 | 0.556 | 0.583 | 0.393 | 0.612 | 0.775 | 0.183 |
| MLP | paper_style | 0.302 | 0.501 | 1.000 | 0.003 | 0.301 | 0.462 | 0.006 | 0.301 | 0.499 | 0.696 |
| RandomForest | fixed_none | 0.765 | 0.649 | 0.361 | 0.938 | 0.720 | 0.477 | 0.383 | 0.650 | 0.791 | 0.163 |
| RandomForest | fixed_smoteenn | 0.688 | 0.706 | 0.750 | 0.661 | 0.490 | 0.591 | 0.380 | 0.605 | 0.778 | 0.204 |
| RandomForest | fixed_smotenc | 0.757 | 0.695 | 0.540 | 0.850 | 0.611 | 0.570 | 0.406 | 0.636 | 0.790 | 0.167 |
| RandomForest | paper_style | 0.729 | 0.696 | 0.614 | 0.778 | 0.551 | 0.577 | 0.384 | 0.605 | 0.773 | 0.180 |
| XGBoost | fixed_none | 0.750 | 0.713 | 0.621 | 0.805 | 0.583 | 0.598 | 0.421 | 0.652 | 0.795 | 0.171 |
| XGBoost | fixed_smoteenn | 0.677 | 0.692 | 0.730 | 0.654 | 0.478 | 0.576 | 0.355 | 0.596 | 0.764 | 0.250 |
| XGBoost | fixed_smotenc | 0.746 | 0.690 | 0.551 | 0.830 | 0.588 | 0.566 | 0.390 | 0.647 | 0.788 | 0.171 |
| XGBoost | paper_style | 0.714 | 0.689 | 0.627 | 0.752 | 0.526 | 0.569 | 0.364 | 0.584 | 0.758 | 0.208 |


*Source: results/all_results.csv*


## Table 4b1: Taiwan Credit — Stage A Model × Variant MCC Matrix

| Model | paper_style | fixed_none | fixed_smotenc | fixed_smoteenn |
| --- | --- | --- | --- | --- |
| LogReg | 0.193 | 0.350 | 0.336 | 0.286 |
| MLP | 0.171 | 0.394 | 0.337 | 0.309 |
| RandomForest | 0.388 | 0.382 | 0.387 | 0.371 |
| XGBoost | 0.398 | 0.391 | 0.408 | 0.385 |


*Source: results/all_results.csv*


## Table 4b2: Taiwan Credit — Stage A Full Benchmark Metrics (Mean)

| Model | Pipeline Variant | Accuracy | Balanced Acc | Recall | Specificity | Precision | F1 | MCC | PR-AUC | ROC-AUC | Brier Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogReg | fixed_none | 0.737 | 0.696 | 0.624 | 0.769 | 0.434 | 0.512 | 0.350 | 0.501 | 0.750 | 0.198 |
| LogReg | fixed_smoteenn | 0.613 | 0.672 | 0.779 | 0.566 | 0.338 | 0.471 | 0.286 | 0.487 | 0.749 | 0.257 |
| LogReg | fixed_smotenc | 0.727 | 0.690 | 0.624 | 0.756 | 0.421 | 0.502 | 0.336 | 0.495 | 0.745 | 0.198 |
| LogReg | paper_style | 0.512 | 0.613 | 0.794 | 0.432 | 0.284 | 0.419 | 0.193 | 0.483 | 0.722 | 0.276 |
| MLP | fixed_none | 0.818 | 0.656 | 0.366 | 0.946 | 0.658 | 0.470 | 0.394 | 0.544 | 0.775 | 0.136 |
| MLP | fixed_smoteenn | 0.675 | 0.683 | 0.698 | 0.668 | 0.374 | 0.487 | 0.309 | 0.503 | 0.746 | 0.255 |
| MLP | fixed_smotenc | 0.730 | 0.689 | 0.616 | 0.762 | 0.425 | 0.502 | 0.337 | 0.507 | 0.750 | 0.189 |
| MLP | paper_style | 0.514 | 0.601 | 0.755 | 0.446 | 0.280 | 0.408 | 0.171 | 0.280 | 0.618 | 0.483 |
| RandomForest | fixed_none | 0.816 | 0.647 | 0.346 | 0.949 | 0.659 | 0.453 | 0.382 | 0.539 | 0.768 | 0.138 |
| RandomForest | fixed_smoteenn | 0.748 | 0.707 | 0.635 | 0.780 | 0.450 | 0.527 | 0.371 | 0.527 | 0.774 | 0.191 |
| RandomForest | fixed_smotenc | 0.797 | 0.686 | 0.487 | 0.885 | 0.545 | 0.514 | 0.387 | 0.531 | 0.763 | 0.154 |
| RandomForest | paper_style | 0.773 | 0.707 | 0.587 | 0.826 | 0.489 | 0.534 | 0.388 | 0.521 | 0.773 | 0.168 |
| XGBoost | fixed_none | 0.762 | 0.715 | 0.631 | 0.799 | 0.471 | 0.539 | 0.391 | 0.562 | 0.784 | 0.177 |
| XGBoost | fixed_smoteenn | 0.754 | 0.714 | 0.641 | 0.786 | 0.461 | 0.536 | 0.385 | 0.548 | 0.779 | 0.186 |
| XGBoost | fixed_smotenc | 0.808 | 0.689 | 0.476 | 0.903 | 0.582 | 0.523 | 0.408 | 0.551 | 0.774 | 0.149 |
| XGBoost | paper_style | 0.783 | 0.707 | 0.570 | 0.844 | 0.510 | 0.538 | 0.398 | 0.538 | 0.776 | 0.159 |


*Source: results/all_results.csv*


## Table 5: Controlled Data Leakage Experiment (Pre-Split vs. In-Fold Resampling)

| Dataset | Model | Leaky F1 | In-Fold F1 | Leaky Recall | In-Fold Recall | Leaky Precision | In-Fold Precision | Leaky MCC | In-Fold MCC | F1 Inflation (Δ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| German | RandomForest | 0.897 | 0.547 | 0.910 | 0.577 | 0.885 | 0.522 | 0.719 | 0.340 | 0.350 |
| Taiwan | RandomForest | 0.936 | 0.534 | 0.925 | 0.591 | 0.947 | 0.488 | 0.833 | 0.388 | 0.402 |


*Source: results/leakage_experiment.csv*


## Table 6: Probability Calibration Audit (Isotonic Regression Brier Reduction)

| Dataset | Model | Variant | Raw Brier Score | Isotonic Calibrated Brier |
| --- | --- | --- | --- | --- |
| German | XGBoost | fixed_none | 0.174 | 0.161 |
| Taiwan | XGBoost | fixed_none | 0.177 | 0.134 |


*Source: results/calibration.csv*


## Table 7: Demographic Fairness & Disparity Audit

| Sensitive Attribute | Demographic Group | Sample Size (N) | Base Default Rate | Selection Rate (Predicted Default) | True Positive Rate (TPR) | False Positive Rate (FPR) |
| --- | --- | --- | --- | --- | --- | --- |
| German: Age (<25) | Age < 25 | 149 | 0.409 | 0.846 | 0.951 | 0.773 |
| German: Age (<25) | Age >= 25 | 851 | 0.281 | 0.632 | 0.891 | 0.531 |
| German: Foreign Worker | Domestic | 37 | 0.108 | 0.324 | 1.000 | 0.242 |
| German: Foreign Worker | Foreign | 963 | 0.307 | 0.677 | 0.902 | 0.577 |
| Taiwan: Sex | Female | 18091 | 0.208 | 0.164 | 0.470 | 0.084 |
| Taiwan: Sex | Male | 11874 | 0.242 | 0.217 | 0.525 | 0.119 |
| Taiwan: Age (<30) | Age < 30 | 9603 | 0.229 | 0.195 | 0.508 | 0.103 |
| Taiwan: Age (<30) | Age >= 30 | 20362 | 0.218 | 0.180 | 0.487 | 0.095 |


*Source: results/fairness_gaps.csv*


## Table 8a: Noise Perturbation Stress Test — German Credit Dataset

| Model | Variant | Noise σ (Std Dev) | MCC Mean | MCC SD | F1 Mean | F1 SD |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | fixed_none | 0.000 | 0.415 | 0.011 | 0.596 | 0.011 |
| XGBoost | fixed_none | 0.100 | 0.377 | 0.045 | 0.561 | 0.033 |
| XGBoost | fixed_none | 0.250 | 0.358 | 0.036 | 0.548 | 0.032 |
| XGBoost | fixed_none | 0.500 | 0.374 | 0.046 | 0.551 | 0.034 |
| XGBoost | fixed_none | 1.000 | 0.364 | 0.055 | 0.550 | 0.047 |
| RandomForest | fixed_none | 0.000 | 0.368 | 0.040 | 0.471 | 0.027 |
| RandomForest | fixed_none | 0.100 | 0.385 | 0.040 | 0.477 | 0.034 |
| RandomForest | fixed_none | 0.250 | 0.363 | 0.051 | 0.458 | 0.039 |
| RandomForest | fixed_none | 0.500 | 0.368 | 0.028 | 0.450 | 0.031 |
| RandomForest | fixed_none | 1.000 | 0.349 | 0.066 | 0.441 | 0.059 |
| LogReg | fixed_none | 0.000 | 0.420 | 0.049 | 0.613 | 0.030 |
| LogReg | fixed_none | 0.100 | 0.413 | 0.050 | 0.608 | 0.030 |
| LogReg | fixed_none | 0.250 | 0.390 | 0.054 | 0.593 | 0.032 |
| LogReg | fixed_none | 0.500 | 0.338 | 0.066 | 0.554 | 0.042 |
| LogReg | fixed_none | 1.000 | 0.333 | 0.033 | 0.542 | 0.024 |


*Source: results/noise_stress_test.csv*


## Table 8b: Noise Perturbation Stress Test — Taiwan Credit Dataset

| Model | Variant | Noise σ (Std Dev) | MCC Mean | MCC SD | F1 Mean | F1 SD |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | fixed_none | 0.000 | 0.388 | 0.009 | 0.538 | 0.007 |
| XGBoost | fixed_none | 0.100 | 0.343 | 0.024 | 0.480 | 0.020 |
| XGBoost | fixed_none | 0.250 | 0.308 | 0.023 | 0.441 | 0.021 |
| XGBoost | fixed_none | 0.500 | 0.274 | 0.029 | 0.409 | 0.027 |
| XGBoost | fixed_none | 1.000 | 0.184 | 0.022 | 0.352 | 0.019 |
| RandomForest | fixed_none | 0.000 | 0.379 | 0.010 | 0.451 | 0.011 |
| RandomForest | fixed_none | 0.100 | 0.332 | 0.011 | 0.354 | 0.010 |
| RandomForest | fixed_none | 0.250 | 0.253 | 0.022 | 0.231 | 0.014 |
| RandomForest | fixed_none | 0.500 | 0.152 | 0.023 | 0.096 | 0.012 |
| RandomForest | fixed_none | 1.000 | 0.064 | 0.006 | 0.019 | 0.004 |
| LogReg | fixed_none | 0.000 | 0.350 | 0.015 | 0.512 | 0.011 |
| LogReg | fixed_none | 0.100 | 0.271 | 0.016 | 0.463 | 0.010 |
| LogReg | fixed_none | 0.250 | 0.202 | 0.013 | 0.424 | 0.006 |
| LogReg | fixed_none | 0.500 | 0.173 | 0.018 | 0.408 | 0.008 |
| LogReg | fixed_none | 1.000 | 0.129 | 0.016 | 0.390 | 0.006 |


*Source: results/noise_stress_test.csv*


## Table 8c: Noise Degradation Summary Matrix (MCC across Noise Levels)

| dataset | model | Noise 0.0 | Noise 0.1 | Noise 0.25 | Noise 0.5 | Noise 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| German | LogReg | 0.420 | 0.413 | 0.390 | 0.338 | 0.333 |
| German | RandomForest | 0.368 | 0.385 | 0.363 | 0.368 | 0.349 |
| German | XGBoost | 0.415 | 0.377 | 0.358 | 0.374 | 0.364 |
| Taiwan | LogReg | 0.350 | 0.271 | 0.202 | 0.173 | 0.129 |
| Taiwan | RandomForest | 0.379 | 0.332 | 0.253 | 0.152 | 0.064 |
| Taiwan | XGBoost | 0.388 | 0.343 | 0.308 | 0.274 | 0.184 |


*Source: results/noise_stress_test.csv*

