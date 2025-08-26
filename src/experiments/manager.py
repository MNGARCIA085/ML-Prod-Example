import os


# get last experiment
def get_last_experiment(base_dir: str) -> str:
    experiments = [f for f in os.listdir(base_dir) if f.startswith("experiment_")]
    if not experiments:
        raise ValueError("No experiments found in directory.")
    experiments.sort()  # timestamp in name ensures chronological order
    return os.path.join(base_dir, experiments[-1])