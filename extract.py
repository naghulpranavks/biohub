import optuna
study = optuna.load_study(study_name="biohub_pp_sweep", storage="sqlite:///optuna_sweep.db")
print(f"BEST SCORE: {study.best_value}")
print("Parameters:")
for k, v in study.best_params.items():
    print(f"os.environ[\"{k}\"] = \"{v}\"")

