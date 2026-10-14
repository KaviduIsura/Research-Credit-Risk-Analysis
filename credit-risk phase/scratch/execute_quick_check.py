import os
import sys
import time
import json
import traceback

def main():
    notebook_path = os.path.abspath("notebooks/Phase2_Credit_Risk_Improved.ipynb")
    print(f"Reading notebook from: {notebook_path}")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    # Verify original cell 4 has QUICK_TEST = False
    cell_4 = nb_data["cells"][4]
    cell_4_text = "".join(cell_4["source"])
    assert "QUICK_TEST = False" in cell_4_text, "Expected QUICK_TEST = False in cell 4"
    
    # Temporarily set QUICK_TEST = True in notebook
    print("Setting QUICK_TEST = True in notebook...")
    cell_4["source"] = [s.replace("QUICK_TEST = False", "QUICK_TEST = True") for s in cell_4["source"]]
    
    # Save notebook with QUICK_TEST = True so kernel/nbconvert executes it
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        ep = ExecutePreprocessor(timeout=600, kernel_name="credit-risk-phase2")
        print("Starting notebook execution with kernel 'credit-risk-phase2'...")
        ep.preprocess(nb, {"metadata": {"path": os.path.abspath("notebooks")}})
        success = True
        print("Notebook execution completed successfully!")
        
        # Save executed notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Execution failed with error:\n{error_msg}")
    finally:
        elapsed = time.time() - start_time
        print(f"Total execution elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        # Always restore QUICK_TEST = False per Step 5
        print("Restoring QUICK_TEST = False in configuration cell...")
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb_restored = json.load(f)
        nb_restored["cells"][4]["source"] = [
            s.replace("QUICK_TEST = True", "QUICK_TEST = False") for s in nb_restored["cells"][4]["source"]
        ]
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb_restored, f, indent=1)
            
        print("Restoration complete. QUICK_TEST is set back to False.")
        
    # Write summary log
    result = {
        "success": success,
        "elapsed_seconds": elapsed,
        "error": error_msg
    }
    with open("scratch/quick_check_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
