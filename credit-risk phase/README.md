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
Open `notebooks/Phase2_Credit_Risk_Improved.ipynb`, select the Python 3 (.venv) kernel.
- Execution mode can be toggled via `QUICK_TEST` in `src/config.py` (True for quick check, False for full run).
- The notebook imports from the modular package in `src/`.
- All metrics and figures are automatically saved to `results/`.

## 5. Project Structure
```
credit-risk phase/
├── data/raw/              # UCI raw datasets (German Credit & Taiwan Default)
├── docs/                  # Coursework descriptors & research paper
├── notebooks/             # Deliverable notebook: Phase2_Credit_Risk_Improved.ipynb
├── src/                   # Modular Python package (loaders, pipelines, metrics, audits)
└── results/               # Generated benchmarks, figures, and CSV tables
```
