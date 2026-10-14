# Credit Risk – Phase 2 (Reproduce the paper's pipeline, fix its flaws, compare)

## 1. Open in Antigravity
1. Install Antigravity from https://antigravity.google/download, sign in with a Google account, choose "Review-driven development".
2. File -> Open Folder -> select THIS folder (`credit-risk-phase2`, not its parent).
3. Open Customizations (Rules / Workflows) and check that `project-rules` and the `/setup-env`, `/run-quick-check`, `/audit-pipeline`, `/summarise-results` workflows are listed.
   (Older Antigravity versions read `.agent/` instead of `.agents/`: if nothing appears, rename the folder.)

## 2. Add the datasets
Follow `data/raw/README.md` (UCI originals, two files into `data/raw/`).

## 3. Python environment (terminal inside Antigravity)
    python -m venv .venv
    # Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    python -m ipykernel install --user --name credit-risk-phase2

## 4. Run the notebook
Open `notebooks/Phase2_Credit_Risk_Improved.ipynb`, select the "credit-risk-phase2" kernel (install the Jupyter extension from the Extensions panel if cells do not run).
- First run with `QUICK_TEST = True` (minutes).
- Then `QUICK_TEST = False`, `DATASETS = ["German"]`, run all.
- Then add `"Taiwan"` (slow: leave it running; finished runs are cached in `results/`).

## 5. Use the agent (Agent Manager, Planning mode)
Type `/` in the agent chat to run a workflow. Useful first prompt:

> Read AGENTS.md and docs/. Run /setup-env, then /run-quick-check. Then run /audit-pipeline and list any risks before I start the full run. Do not change code without asking.
