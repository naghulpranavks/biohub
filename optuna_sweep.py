import os
import subprocess
import optuna
import re
from pathlib import Path

# The downloaded Kaggle competition training data folder
DATA_DIR = r"D:\Downloads_2\biohub-cell-tracking-during-development\train"
# The project root where the 'repo' directory was extracted
PROJECT_DIR = r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking"

# We will use a few small validation videos to keep the loop fast but robust.
VALIDATION_VIDEOS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_0c582fdc",
    "44b6_0db75fae",
    "44b6_12dfb391"
]

def run_local_validation(trial):
    """
    Runs the inference pipeline on local validation data and returns the score.
    """
    # 1. Define the hyperparameter search space for this trial
    # Using the env vars that the Kaggle notebook sweeps.
    env = os.environ.copy()
    
    # Add repo/src to PYTHONPATH so it can find biohub_tracking
    repo_src_path = os.path.join(PROJECT_DIR, "repo", "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src_path};{existing_pythonpath}" if existing_pythonpath else repo_src_path
    
    env["BIOHUB_GAP_CLOSE_MAX_GAP"] = str(trial.suggest_int("gap_max_gap", 1, 5))
    env["BIOHUB_GAP_CLOSE_UM"] = str(trial.suggest_float("gap_close_um", 2.0, 10.0))
    
    # Gap density adaptivity variables
    env["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
    env["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = str(trial.suggest_float("gap_ref_um", 3.0, 9.0))
    env["BIOHUB_GAP_DENSITY_GAIN"] = str(trial.suggest_float("gap_gain", 0.01, 0.10))
    env["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = str(trial.suggest_float("gap_max_step", 0.05, 0.30))
    env["BIOHUB_GAP_DENSITY_NEIGHBORS"] = str(trial.suggest_int("gap_neighbors", 1, 5))
    
    # Short track filtering
    env["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
    env["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = str(trial.suggest_int("min_track_len", 3, 10))
    
    # We set BIOHUB_DATA_DIR so dataspec.py looks in our downloaded folder
    env["BIOHUB_DATA_DIR"] = DATA_DIR
    
    proxy_score = 0.0
    
    for video_name in VALIDATION_VIDEOS:
        # Pass the path without .zarr so predict_unet_transformer uses the base name 
        # and saves as {base_name}.geff. open_dataset handles appending .zarr if needed.
        video_path_no_ext = os.path.join(DATA_DIR, video_name)
        print(f"\n[Trial {trial.number}] Running inference on {video_name}...")
        
        # Use the virtual environment Python
        python_exe = os.path.join(PROJECT_DIR, "venv", "Scripts", "python.exe")
        
        # Run predict_unet_transformer.py
        cmd_predict = [
            python_exe, "repo/scripts/predict_unet_transformer.py",
            "--debug-video", video_path_no_ext,
            "--use-ilp"
        ]
        
        res_predict = subprocess.run(cmd_predict, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
        if res_predict.returncode != 0:
            print(f"Error during prediction:\n{res_predict.stderr}")
            raise optuna.TrialPruned()
            
        # Run evaluate.py to get the proxy score (Adjusted Edge Jaccard)
        print(f"[Trial {trial.number}] Evaluating {video_name}...")
        cmd_evaluate = [
            python_exe, "repo/scripts/evaluate.py"
        ]
        res_evaluate = subprocess.run(cmd_evaluate, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
        
        if res_evaluate.returncode != 0:
            print(f"Error during evaluation:\n{res_evaluate.stderr}")
            raise optuna.TrialPruned()
            
        # Parse the output string from evaluate.py.
        # Example output line:
        #   naghulpranavks/unet_transformer/0: score=0.9100  edge_jaccard=0.9200  adj_edge_jaccard=0.9234 ...
        match = re.search(r"adj_edge_jaccard=([0-9.]+)", res_evaluate.stdout)
        if match:
            score = float(match.group(1))
            proxy_score += score
        else:
            print("Could not parse adj_edge_jaccard from evaluate.py output:")
            print(res_evaluate.stdout)
            raise optuna.TrialPruned()
            
    # Average the score across all validation videos
    final_score = proxy_score / len(VALIDATION_VIDEOS)
    print(f"Trial {trial.number} Final Score: {final_score:.4f}")
    
    return final_score

def main():
    print(f"Starting Optuna Hyperparameter Sweep...")
    print(f"Validation Videos: {VALIDATION_VIDEOS}")
    
    # Create the Optuna study aiming to MAXIMIZE the score
    study = optuna.create_study(
        study_name="biohub_pp_sweep",
        direction="maximize",
        storage="sqlite:///optuna_sweep.db",
        load_if_exists=True
    )
    
    # Run the optimization loop
    study.optimize(run_local_validation, n_trials=3)
    
    print("\nSweep Complete!")
    print(f"Best Score: {study.best_value}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
