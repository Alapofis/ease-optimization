import argparse

from common import (
    append_row,
    apply_dry_run_limits,
    ensure_output_dir,
    fit_and_evaluate,
    load_base_config,
    load_split,
    load_yaml,
    prepare_run_dir,
    single_threaded,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/block_sweep_m2.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sweep_cfg = load_yaml(args.config)
    base = load_base_config(sweep_cfg["base_config"])

    if args.dry_run:
        apply_dry_run_limits(base)
        sweep_cfg["sweep"]["block_size"] = [1, 4]
        sweep_cfg["fixed"]["nystrom_rank"] = 16
        sweep_cfg["fixed"]["nystrom_oversample"] = 5

    if args.run_dir:
        base["runtime"]["output_dir"] = str(
            prepare_run_dir(base, run_dir=args.run_dir, overwrite=args.overwrite)
        )

    X_train, X_valid, _ = load_split(base)
    out_path = ensure_output_dir(base) / "block_sweep.csv"

    for block_size in sweep_cfg["sweep"]["block_size"]:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_valid,
            epsilon=float(sweep_cfg["fixed"]["epsilon"]),
            block_size=int(block_size),
            nystrom_rank=int(sweep_cfg["fixed"]["nystrom_rank"]),
            nystrom_oversample=int(sweep_cfg["fixed"]["nystrom_oversample"]),
            show_progress=True,
        )
        row = {"block_size": int(block_size), **row}
        append_row(out_path, row)
        print(row)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    with single_threaded():
        main()
