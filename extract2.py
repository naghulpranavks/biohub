import optuna
study = optuna.load_study(study_name="biohub_pp_sweep", storage="sqlite:///optuna_sweep.db")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number > 104]
if completed:
    best_5v = max(completed, key=lambda t: t.value)
    print(f"Best 5-video score: {best_5v.value}")
    print("Parameters:")
    for k, v in best_5v.params.items():
        print(f"os.environ[\"{k}\"] = \"{v}\"")

