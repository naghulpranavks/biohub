import optuna
study = optuna.load_study(study_name="biohub_pp_sweep", storage="sqlite:///optuna_sweep.db")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number > 206]
for t in completed:
    print(f"Trial {t.number}: {t.value:.4f}")

