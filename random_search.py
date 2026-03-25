"""
Random hyperparameter search for the baseline SGD experiment.
"""
import argparse
import json
import math
import random

from main import DEFAULT_CONFIG, run_baseline


SEARCH_SPACE = {
    "weight_decay": (1e-5, 1e-2),
    "momentum": (0.8, 0.99),
    "lr": (1e-3, 3e-1),
    "batch_size": [64, 128, 256, 512],
    "epochs": [25, 50, 100, 150, 200],
}


def sample_log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(rng, base_config):
    config = dict(base_config)
    config["weight_decay"] = sample_log_uniform(rng, *SEARCH_SPACE["weight_decay"])
    config["momentum"] = rng.uniform(*SEARCH_SPACE["momentum"])
    config["lr"] = sample_log_uniform(rng, *SEARCH_SPACE["lr"])
    config["batch_size"] = rng.choice(SEARCH_SPACE["batch_size"])
    config["epochs"] = rng.choice(SEARCH_SPACE["epochs"])
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run random search over baseline training hyperparameters."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_CONFIG["train_fraction"]
    )
    parser.add_argument("--val-split", type=float, default=DEFAULT_CONFIG["val_split"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument(
        "--output",
        default="random_search_results.json",
        help="Path to save all trial results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    base_config = {
        **DEFAULT_CONFIG,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_split": args.val_split,
        "device": args.device,
    }

    results = []
    best_result = None

    for trial in range(1, args.trials + 1):
        trial_config = sample_config(rng, base_config)

        print(f"\n=== Trial {trial}/{args.trials} ===")
        print(
            "Config: "
            f"lr={trial_config['lr']:.6f}, "
            f"momentum={trial_config['momentum']:.4f}, "
            f"weight_decay={trial_config['weight_decay']:.6f}, "
            f"batch_size={trial_config['batch_size']}, "
            f"epochs={trial_config['epochs']}"
        )

        history = run_baseline(trial_config)
        best_val_acc = max(history["val_acc"])

        trial_result = {
            "trial": trial,
            "config": trial_config,
            "best_val_acc": best_val_acc,
            "final_val_acc": history["val_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
        }
        results.append(trial_result)

        if best_result is None or best_val_acc > best_result["best_val_acc"]:
            best_result = trial_result

        print(f"Trial best val accuracy: {best_val_acc:.4f}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "search_seed": args.seed,
                "trials": results,
                "best_result": best_result,
            },
            handle,
            indent=2,
        )

    print("\n=== Random Search Summary ===")
    print(f"Results saved to: {args.output}")
    print(
        "Best config: "
        f"lr={best_result['config']['lr']:.6f}, "
        f"momentum={best_result['config']['momentum']:.4f}, "
        f"weight_decay={best_result['config']['weight_decay']:.6f}, "
        f"batch_size={best_result['config']['batch_size']}, "
        f"epochs={best_result['config']['epochs']}"
    )
    print(f"Best validation accuracy: {best_result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
