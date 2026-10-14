# Project rules (always apply)

- Never fabricate, estimate or "round up" experimental results. Only report numbers printed by code that actually ran.
- Prevent data leakage: every preprocessing, resampling or tuning step must live inside the sklearn/imblearn Pipeline so it is fit on training folds only.
- Use SEED = 42 everywhere; keep results reproducible.
- Never modify or delete anything in `docs/` or `data/raw/`.
- Do not change the pre-declared `PRIMARY` pipeline after seeing results. If you disagree with it, say so in the summary instead.
- Keep authors' numbers (F1 0.937 German, 0.610 Taiwan) as given; do not "adjust" them.
- Prefer small, reviewable edits to the notebook over rewrites; explain each change in one sentence.
- Run long jobs (Taiwan nested CV) only when I ask; start with QUICK_TEST = True.
- Be honest about limitations (different folds from the authors, simple sklearn MLP, reduced Taiwan tuning).
