from __future__ import annotations

from pathlib import Path

from src.experiments.run_entropy_guided_experiments import ExperimentConfig, run_experiments


def main() -> None:
    output_dir = Path("src/outputs")
    config = ExperimentConfig()
    run_experiments(config=config, output_dir=output_dir)
    print(f"Saved experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()