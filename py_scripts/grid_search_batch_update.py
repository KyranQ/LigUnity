"""Grid search helper for batch size and update frequency.

This script runs `unicore-train` multiple times while sweeping
`--batch-size` and `--update-freq`. It mirrors the training options
used in `train.sh` and keeps outputs for each combination in separate
folders under the chosen save root.

You can launch it as a standalone utility, e.g.:

```
python py_scripts/grid_search_batch_update.py ./data/ --batch-sizes 16,32 --update-freqs 1,2 --run-name bs_uf_search
```

Each run stores logs and a `config.json` that records the exact command
and effective batch size so you can trace results later.
"""

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from typing import Iterable, List


DEFAULT_CMD_ARGS = [
    "--train-subset",
    "train",
    "--valid-subset",
    "valid",
    "--num-workers",
    "8",
    "--ddp-backend",
    "c10d",
    "--task",
    "train_task",
    "--loss",
    "rank_softmax",
    "--arch",
    "pocketscreen",
    "--max-pocket-atoms",
    "256",
    "--optimizer",
    "adam",
    "--adam-betas",
    "(0.9, 0.999)",
    "--adam-eps",
    "1e-8",
    "--clip-norm",
    "1.0",
    "--lr-scheduler",
    "polynomial_decay",
    "--validate-interval",
    "1",
    "--best-checkpoint-metric",
    "valid_bedroc",
    "--patience",
    "2000",
    "--all-gather-list-size",
    "2048000",
    "--keep-best-checkpoints",
    "8",
    "--keep-last-epochs",
    "10",
    "--find-unused-parameters",
    "--maximize-best-checkpoint-metric",
]


def parse_list(value: str) -> List[int]:
    try:
        return [int(v) for v in value.split(",")]
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise argparse.ArgumentTypeError("List must be comma-separated integers") from exc


def build_command(args: argparse.Namespace, batch_size: int, update_freq: int, save_dir: str, tmp_save_dir: str, tsb_dir: str) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        str(args.n_gpu),
        "--master_port",
        str(args.master_port),
        "unicore-train",
        args.data,
        "--user-dir",
        "./unimol",
        *DEFAULT_CMD_ARGS,
        "--lr",
        str(args.lr),
        "--warmup-ratio",
        str(args.warmup),
        "--max-epoch",
        str(args.max_epoch),
        "--batch-size",
        str(batch_size),
        "--batch-size-valid",
        str(args.batch_size_valid),
        "--update-freq",
        str(update_freq),
        "--seed",
        str(args.seed),
        "--tensorboard-logdir",
        tsb_dir,
        "--log-interval",
        str(args.log_interval),
        "--log-format",
        "simple",
        "--save-dir",
        save_dir,
        "--tmp-save-dir",
        tmp_save_dir,
        "--finetune-pocket-model",
        args.finetune_pocket_model,
        "--finetune-mol-model",
        args.finetune_mol_model,
        "--valid-set",
        args.valid_set,
        "--max-lignum",
        str(args.max_lignum),
        "--protein-similarity-thres",
        str(args.protein_similarity_thres),
    ]

    if args.fp16:
        cmd.extend(["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"])

    if args.rank_weight is not None:
        cmd.extend(["--rank-weight", str(args.rank_weight)])

    if args.dropout is not None:
        cmd.extend(["--dropout", str(args.dropout)])

    if args.dist_threshold is not None:
        cmd.extend(["--dist-threshold", str(args.dist_threshold)])

    if args.recycling is not None:
        cmd.extend(["--recycling", str(args.recycling)])

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    return cmd


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def write_metadata(metadata_path: str, command: List[str], batch_size: int, update_freq: int) -> None:
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_size": batch_size,
                "update_freq": update_freq,
                "effective_batch_per_gpu": batch_size * update_freq,
                "command": command,
            },
            f,
            indent=2,
        )


def run_grid_search(args: argparse.Namespace) -> None:
    combinations: Iterable[tuple[int, int]] = itertools.product(args.batch_sizes, args.update_freqs)
    log_root = os.path.join(args.save_root, "train_log")
    ensure_dirs(log_root)

    for batch_size, update_freq in combinations:
        tag = f"{args.run_name}_bs{batch_size}_uf{update_freq}"
        run_root = os.path.join(args.save_root, tag)
        save_dir = os.path.join(run_root, "savedir_screen")
        tmp_save_dir = os.path.join(run_root, "tmp_save_dir_screen")
        tsb_dir = os.path.join(run_root, "tsb_dir_screen")
        ensure_dirs(save_dir, tmp_save_dir, tsb_dir)

        cmd = build_command(args, batch_size, update_freq, save_dir, tmp_save_dir, tsb_dir)

        metadata_path = os.path.join(run_root, "config.json")
        write_metadata(metadata_path, cmd, batch_size, update_freq)

        log_path = os.path.join(log_root, f"train_log_{tag}.txt")
        effective_batch = batch_size * update_freq
        print(
            "[GridSearch] Running batch_size=%s, update_freq=%s (effective per-GPU batch=%s)"
            % (batch_size, update_freq, effective_batch)
        )
        print(f"             Logs: {log_path}")
        print(f"             Save dir: {run_root}")
        if args.dry_run:
            print("             Dry-run enabled, command not executed.")
            print("             Command:", " ".join(cmd))
            continue

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            print(f"[GridSearch] Combination batch_size={batch_size}, update_freq={update_freq} failed with code {process.returncode}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search batch_size and update_freq for LigUnity training.")
    parser.add_argument("data", help="Downstream data path for unicore-train.")
    parser.add_argument("--batch-sizes", type=parse_list, default=[24], help="Comma-separated batch sizes to try (per GPU).")
    parser.add_argument("--update-freqs", type=parse_list, default=[1], help="Comma-separated update_freq values to try.")
    parser.add_argument("--save-root", default="./save", help="Root directory for saving runs and logs.")
    parser.add_argument("--run-name", default="grid_search", help="Prefix name for this grid search run.")
    parser.add_argument("--n-gpu", type=int, default=2, help="Number of GPUs for torch.distributed.launch.")
    parser.add_argument("--master-port", type=int, default=10062, help="Master port for distributed training.")
    parser.add_argument("--max-epoch", type=int, default=50, help="Maximum number of epochs.")
    parser.add_argument("--batch-size-valid", type=int, default=32, help="Validation batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.06, help="Warmup ratio.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval for training.")
    parser.add_argument("--finetune-mol-model", default="./pretrain/mol_pre_no_h_220816.pt", help="Path to pretrained molecular model.")
    parser.add_argument("--finetune-pocket-model", default="./pretrain/pocket_pre_220816.pt", help="Path to pretrained pocket model.")
    parser.add_argument("--valid-set", default="CASF", help="Validation set name.")
    parser.add_argument("--max-lignum", type=int, default=16, help="Maximum ligand number.")
    parser.add_argument("--protein-similarity-thres", type=float, default=1.0, help="Protein similarity threshold.")
    parser.add_argument("--rank-weight", type=float, default=None, help="Optional rank weight; omit to use default.")
    parser.add_argument("--dropout", type=float, default=None, help="Optional dropout rate override.")
    parser.add_argument("--dist-threshold", type=float, default=None, help="Optional distance threshold override.")
    parser.add_argument("--recycling", type=int, default=None, help="Optional recycling iterations override.")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed-precision training (enabled by default).")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable mixed-precision training.")
    parser.add_argument("--extra-args", default="", help="Extra arguments appended to the unicore-train command.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    args = parser.parse_args()
    run_grid_search(args)


if __name__ == "__main__":
    main()
