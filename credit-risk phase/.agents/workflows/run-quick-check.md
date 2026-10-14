---
description: Dry-run the whole notebook quickly to prove it works
---
1. Open `notebooks/Phase2_Credit_Risk_Improved.ipynb`.
2. Set `QUICK_TEST = True` in the configuration cell (do not commit this value).
3. Run all cells top to bottom. If a cell fails, fix the smallest possible cause and rerun from that cell.
4. Report: runtime, any warnings, and confirm that cleaning counts match: Taiwan duplicates 35, EDUCATION codes 0/5/6 = 345 rows, MARRIAGE code 0 = 54 rows (full data only).
5. Set `QUICK_TEST = False` again. Do not interpret the quick-test numbers.
