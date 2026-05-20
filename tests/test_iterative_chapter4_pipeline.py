import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from common import write_manifest
from make_chapter4_tables import make_tables
from select_config import select_block, select_epsilon, select_rank


def test_selection_logic():
    epsilon_df = pd.DataFrame(
        [
            {"epsilon": 1e-2, "t_solve": 1.0, "recall": 0.898, "ndcg": 0.796, "q_epsilon": 0.99},
            {"epsilon": 1e-3, "t_solve": 2.0, "recall": 0.900, "ndcg": 0.800, "q_epsilon": 1.00},
        ]
    )
    selection = {"tau_recall": 0.005, "tau_ndcg": 0.005, "q_min": 0.95}
    assert select_epsilon(epsilon_df, selection)["epsilon"] == 1e-2

    rank_df = pd.DataFrame(
        [
            {"nystrom_rank": 0, "t_total": 5.0, "t_solve": 5.0, "mean_iterations": 10, "q_epsilon": 1.0},
            {"nystrom_rank": 32, "t_total": 3.0, "t_solve": 2.5, "mean_iterations": 6, "q_epsilon": 1.0},
        ]
    )
    assert select_rank(rank_df, selection)["nystrom_rank"] == 32

    block_df = pd.DataFrame(
        [
            {"block_size": 1, "t_solve": 4.0, "memory_gb": 1.0, "recall": 0.90, "ndcg": 0.80, "q_epsilon": 1.0},
            {"block_size": 4, "t_solve": 2.0, "memory_gb": 1.2, "recall": 0.899, "ndcg": 0.799, "q_epsilon": 1.0},
        ]
    )
    assert select_block(block_df, selection, max_memory_gb=2.0)["block_size"] == 4


def test_manifest_writer(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_manifest(
        path,
        config={"model": {"reg": 500.0}},
        dataset_stats={"n_users": 10, "n_items": 5, "nnz": 20},
        selected={"epsilon": {"epsilon": 1e-3}},
        memory_estimate={"estimated_total_gb": 0.1},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["model"]["reg"] == 500.0
    assert data["dataset"]["n_users"] == 10
    assert "numpy" in data["libraries"]


def test_make_chapter4_tables(tmp_path):
    run_dir = tmp_path
    write_manifest(
        run_dir / "run_manifest.json",
        config={},
        dataset_stats={
            "n_users": 10,
            "n_items": 5,
            "nnz": 20,
            "density": 0.4,
            "mean_items_per_user": 2.0,
            "mean_users_per_item": 4.0,
        },
    )
    pd.DataFrame(
        [{"epsilon": 1e-3, "t_solve": 1.0, "mean_iterations": 2.0, "q_epsilon": 1.0, "recall": 0.5, "ndcg": 0.4}]
    ).to_csv(run_dir / "epsilon_sweep.csv", index=False)
    pd.DataFrame(
        [{"nystrom_rank": 0, "t_build": 0.0, "t_solve": 1.0, "t_total": 1.0, "mean_iterations": 2.0, "memory_gb": 0.1}]
    ).to_csv(run_dir / "rank_sweep.csv", index=False)
    pd.DataFrame(
        [{"block_size": 1, "t_solve": 1.0, "mean_iterations": 2.0, "q_epsilon": 1.0, "memory_gb": 0.1, "ndcg": 0.4}]
    ).to_csv(run_dir / "block_sweep.csv", index=False)
    pd.DataFrame(
        [{"method": "Block PCG", "t_total": 1.0, "memory_gb": 0.1, "q_epsilon": 1.0, "recall": 0.5, "ndcg": 0.4}]
    ).to_csv(run_dir / "internal_ease_baselines.csv", index=False)
    pd.DataFrame(
        [{"method": "Popularity", "status": "ok", "t_fit": 0.01, "recall": 0.2, "ndcg": 0.1}]
    ).to_csv(run_dir / "external_baselines.csv", index=False)
    (run_dir / "figures").mkdir()
    for name in [
        "epsilon_tradeoff.png",
        "rank_time_iterations.png",
        "block_time_memory.png",
        "internal_ease_baselines.png",
        "external_baselines_quality.png",
    ]:
        (run_dir / "figures" / name).write_bytes(b"png")
    (run_dir / "selected_config.yaml").write_text(
        yaml.safe_dump(
            {
                "selected": {
                    "epsilon": {"epsilon": 1e-3, "selected_recall": 0.5, "selected_ndcg": 0.4, "selected_q_epsilon": 1.0},
                    "rank": {"nystrom_rank": 16, "selected_t_total": 1.0},
                    "block": {"block_size": 4, "selected_t_solve": 0.8, "selected_t_total": 0.9, "selected_memory_gb": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )

    tables_dir = make_tables(run_dir)

    assert (tables_dir / "dataset_table.tex").exists()
    epsilon_table = (tables_dir / "epsilon_sweep_table.tex").read_text(encoding="utf-8")
    assert "Recall@$K$" in epsilon_table
    assert "$10^{-3}$" in epsilon_table
    assert "\\caption" in epsilon_table
    assert "\\label" in epsilon_table

    selected_table = (tables_dir / "selected_parameters_table.tex").read_text(encoding="utf-8")
    assert "$\\varepsilon^*$" in selected_table
    assert "$k^*$" in selected_table
    assert "$b^*$" in selected_table

    figures_tex = (run_dir / "chapter4_figures.tex").read_text(encoding="utf-8")
    assert "\\includegraphics" in figures_tex
    for rel_path in re.findall(r"\\includegraphics\\[.*?\\]\\{(.+?)\\}", figures_tex):
        assert (run_dir / rel_path).exists()

    summary = (run_dir / "chapter4_summary.md").read_text(encoding="utf-8")
    assert "selected_parameters_table.tex" in summary


def test_run_chapter4_dry_run_creates_artifacts(tmp_path):
    cmd = [
        sys.executable,
        str(EXPERIMENTS / "run_chapter4.py"),
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    assert "Chapter 4 artifacts are ready" in result.stdout

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for name in [
        "epsilon_sweep.csv",
        "rank_sweep.csv",
        "block_sweep.csv",
        "internal_ease_baselines.csv",
        "external_baselines.csv",
        "selected_config.yaml",
        "run_manifest.json",
        "chapter4_figures.tex",
        "chapter4_summary.md",
    ]:
        assert (run_dir / name).exists()

    assert (run_dir / "figures" / "epsilon_tradeoff.png").exists()
    assert (run_dir / "figures" / "internal_ease_baselines.png").exists()
    assert (run_dir / "tables" / "external_baselines_table.tex").exists()
    assert (run_dir / "tables" / "selected_parameters_table.tex").exists()

    selected_table = (run_dir / "tables" / "selected_parameters_table.tex").read_text(encoding="utf-8")
    assert "$\\varepsilon^*$" in selected_table
    assert "$k^*$" in selected_table
    assert "$b^*$" in selected_table
