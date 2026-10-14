---
description: Write results/SUMMARY.md from the finished notebook outputs
---
1. Read the CSV files in `results/` (all_results.csv, verdicts.csv, fix_vs_paper_style_tests.csv, leakage_experiment.csv, noise_stress_test.csv, fairness_gaps.csv, calibration.csv).
2. Write `results/SUMMARY.md` with: (a) effect of the fixes vs paper_style, (b) verdict vs authors per dataset, (c) leakage experiment, (d) calibration, (e) fairness gaps, (f) noise test, (g) limitations.
3. Quote only numbers present in the CSV files. If a file is missing, say so instead of guessing.
