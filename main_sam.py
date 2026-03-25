"""
Run the SAM experiment on CIFAR-10.
"""
import argparse

from src.model_sam import SAM_CONFIG, run_sam


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SAM CIFAR-10 experiment.")
    parser.add_argument("--seed", type=int, default=SAM_CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=SAM_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=SAM_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=SAM_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=SAM_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=SAM_CONFIG["weight_decay"])
    parser.add_argument("--rho", type=float, default=SAM_CONFIG["rho"])
    parser.add_argument(
        "--train-fraction", type=float, default=SAM_CONFIG["train_fraction"]
    )
    parser.add_argument("--val-split", type=float, default=SAM_CONFIG["val_split"])
    parser.add_argument("--device", default=SAM_CONFIG["device"])
    return parser.parse_args()


def main():
    args = parse_args()
    run_sam(
        {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "rho": args.rho,
            "train_fraction": args.train_fraction,
            "val_split": args.val_split,
            "device": args.device,
        }
    )


if __name__ == "__main__":
    main()
