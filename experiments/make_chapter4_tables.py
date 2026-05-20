import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import yaml


FIGURES = [
    (
        "4.2",
        "figures/epsilon_tradeoff.png",
        "Влияние порога невязки $\\varepsilon$ на время решения и качество рекомендаций",
        "fig:epsilon_tradeoff",
    ),
    (
        "4.3",
        "figures/rank_time_iterations.png",
        "Влияние ранга предобуславливателя на время решения и число итераций",
        "fig:rank_time_iterations",
    ),
    (
        "4.4",
        "figures/block_time_memory.png",
        "Влияние ширины блока на время решения и потребление памяти",
        "fig:block_time_memory",
    ),
    (
        "4.5",
        "figures/internal_ease_baselines.png",
        "Сравнение внутренних вариантов итерационного обучения EASE",
        "fig:internal_ease_baselines",
    ),
    (
        "4.5",
        "figures/external_baselines_quality.png",
        "Сравнение качества итоговой конфигурации с baseline-алгоритмами",
        "fig:external_baselines_quality",
    ),
]


REQUIRED_ARTIFACTS = [
    "epsilon_sweep.csv",
    "rank_sweep.csv",
    "block_sweep.csv",
    "internal_ease_baselines.csv",
    "external_baselines.csv",
    "selected_config.yaml",
    "run_manifest.json",
    "figures/epsilon_tradeoff.png",
    "figures/rank_time_iterations.png",
    "figures/block_time_memory.png",
    "figures/internal_ease_baselines.png",
    "figures/external_baselines_quality.png",
    "tables/dataset_table.tex",
    "tables/environment_table.tex",
    "tables/epsilon_sweep_table.tex",
    "tables/rank_sweep_table.tex",
    "tables/block_sweep_table.tex",
    "tables/internal_baselines_table.tex",
    "tables/external_baselines_table.tex",
    "tables/selected_parameters_table.tex",
    "chapter4_figures.tex",
    "chapter4_summary.md",
]


def _fmt_epsilon(value) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value <= 0:
        return str(value)
    exponent = int(round(math.log10(value)))
    if math.isclose(value, 10**exponent, rel_tol=1e-9, abs_tol=1e-12):
        return f"$10^{{{exponent}}}$"
    return f"${value:.1e}$"


def _fmt(value, kind: str | None = None):
    if pd.isna(value):
        return "--"
    if kind == "epsilon":
        return _fmt_epsilon(value)
    if isinstance(value, float):
        if kind in {"time", "memory", "iterations"}:
            return f"{value:.3f}"
        if kind in {"quality", "ratio"}:
            return f"{value:.4f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def _latex_table(headers, rows, caption, label, colspec=None, kinds=None):
    if colspec is None:
        colspec = "|" + "|".join(["c"] * len(headers)) + "|"
    kinds = kinds or [None] * len(headers)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + colspec + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_fmt(v, kind) for v, kind in zip(row, kinds)) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"saved {path}")


def make_dataset_table(run_dir: Path, tables_dir: Path):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = manifest.get("dataset", {})
    rows = [
        ["Число пользователей $M$", data.get("n_users")],
        ["Число объектов $N$", data.get("n_items")],
        ["Число взаимодействий $\\operatorname{nnz}(X)$", data.get("nnz")],
        ["Плотность матрицы", data.get("density")],
        ["Среднее число взаимодействий на пользователя", data.get("mean_items_per_user")],
        ["Среднее число взаимодействий на объект", data.get("mean_users_per_item")],
    ]
    _write(
        tables_dir / "dataset_table.tex",
        _latex_table(
            ["Показатель", "Значение"],
            rows,
            "Характеристики экспериментального набора данных",
            "tab:dataset",
            colspec="|p{0.62\\textwidth}|c|",
        ),
    )


def make_environment_table(run_dir: Path, tables_dir: Path):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform = manifest.get("platform", {})
    libs = manifest.get("libraries", {})
    thread_env = manifest.get("thread_env", {})
    rows = [
        ["CPU/Platform", platform.get("system")],
        ["Python", platform.get("python")],
        ["NumPy", libs.get("numpy")],
        ["SciPy", libs.get("scipy")],
        ["Pandas", libs.get("pandas")],
        ["Формат хранения $X$", "CSR"],
        ["Число потоков BLAS", thread_env.get("OPENBLAS_NUM_THREADS") or "1"],
    ]
    _write(
        tables_dir / "environment_table.tex",
        _latex_table(
            ["Параметр", "Значение"],
            rows,
            "Аппаратная и программная среда экспериментов",
            "tab:environment",
            colspec="|p{0.34\\textwidth}|p{0.54\\textwidth}|",
        ),
    )


def _csv_table(
    run_dir: Path,
    tables_dir: Path,
    filename: str,
    output: str,
    headers,
    columns,
    caption,
    label,
    colspec=None,
    kinds=None,
):
    path = run_dir / filename
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = df[columns].values.tolist()
    _write(tables_dir / output, _latex_table(headers, rows, caption, label, colspec=colspec, kinds=kinds))


def make_result_tables(run_dir: Path, tables_dir: Path):
    _csv_table(
        run_dir,
        tables_dir,
        "epsilon_sweep.csv",
        "epsilon_sweep_table.tex",
        ["$\\varepsilon$", "$T_{solve}$, c", "$\\bar n_{it}$", "$q_\\varepsilon$", "Recall@$K$", "NDCG@$K$"],
        ["epsilon", "t_solve", "mean_iterations", "q_epsilon", "recall", "ndcg"],
        "Влияние порога невязки $\\varepsilon$ на время решения и качество рекомендаций",
        "tab:epsilon_sweep",
        kinds=["epsilon", "time", "iterations", "ratio", "quality", "quality"],
    )
    _csv_table(
        run_dir,
        tables_dir,
        "rank_sweep.csv",
        "rank_sweep_table.tex",
        ["$k$", "$T_{build}$, c", "$T_{solve}$, c", "$T_{total}$, c", "$\\bar n_{it}$", "Память, ГБ"],
        ["nystrom_rank", "t_build", "t_solve", "t_total", "mean_iterations", "memory_gb"],
        "Влияние ранга предобуславливателя $k$ на время и сходимость",
        "tab:rank_sweep",
        kinds=[None, "time", "time", "time", "iterations", "memory"],
    )
    _csv_table(
        run_dir,
        tables_dir,
        "block_sweep.csv",
        "block_sweep_table.tex",
        ["$b$", "$T_{solve}$, c", "$\\bar n_{it}$", "$q_\\varepsilon$", "Память, ГБ", "NDCG@$K$"],
        ["block_size", "t_solve", "mean_iterations", "q_epsilon", "memory_gb", "ndcg"],
        "Влияние ширины блока $b$ на время решения и память",
        "tab:block_sweep",
        kinds=[None, "time", "iterations", "ratio", "memory", "quality"],
    )
    _csv_table(
        run_dir,
        tables_dir,
        "internal_ease_baselines.csv",
        "internal_baselines_table.tex",
        ["Метод", "Время, c", "Память, ГБ", "$q_\\varepsilon$", "Recall@$K$", "NDCG@$K$"],
        ["method", "t_total", "memory_gb", "q_epsilon", "recall", "ndcg"],
        "Сравнение вариантов итерационного обучения EASE",
        "tab:internal_baselines",
        colspec="|p{0.38\\textwidth}|c|c|c|c|c|",
        kinds=[None, "time", "memory", "ratio", "quality", "quality"],
    )
    _csv_table(
        run_dir,
        tables_dir,
        "external_baselines.csv",
        "external_baselines_table.tex",
        ["Метод", "Статус", "Время обучения, c", "Recall@$K$", "NDCG@$K$"],
        ["method", "status", "t_fit", "recall", "ndcg"],
        "Сравнение итоговой конфигурации EASE с baseline-алгоритмами",
        "tab:external_baselines",
        colspec="|p{0.34\\textwidth}|c|c|c|c|",
        kinds=[None, None, "time", "quality", "quality"],
    )


def make_selected_parameters_table(run_dir: Path, tables_dir: Path):
    config_path = run_dir / "selected_config.yaml"
    if not config_path.exists():
        return
    selected_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = selected_config.get("selected", {})
    epsilon = selected.get("epsilon", {})
    rank = selected.get("rank", {})
    block = selected.get("block", {})
    rows = [
        ["$\\varepsilon^*$", _fmt(epsilon.get("epsilon"), "epsilon")],
        ["$k^*$", _fmt(rank.get("nystrom_rank"))],
        ["$b^*$", _fmt(block.get("block_size"))],
        ["Recall@$K$", _fmt(block.get("selected_recall", epsilon.get("selected_recall")), "quality")],
        ["NDCG@$K$", _fmt(block.get("selected_ndcg", epsilon.get("selected_ndcg")), "quality")],
        ["$q_\\varepsilon$", _fmt(block.get("selected_q_epsilon", epsilon.get("selected_q_epsilon")), "ratio")],
        ["$T_{solve}$, c", _fmt(block.get("selected_t_solve", epsilon.get("selected_t_solve")), "time")],
        ["$T_{total}$, c", _fmt(block.get("selected_t_total", rank.get("selected_t_total")), "time")],
        ["Память, ГБ", _fmt(block.get("selected_memory_gb"), "memory")],
    ]
    _write(
        tables_dir / "selected_parameters_table.tex",
        _latex_table(
            ["Параметр", "Значение"],
            rows,
            "Выбранная конфигурация приближённого обучения EASE",
            "tab:selected_parameters",
            colspec="|p{0.42\\textwidth}|c|",
            kinds=[None, None],
        )
    )


def _figure_block(path: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            "\\begin{figure}[H]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{path}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{figure}",
            "",
        ]
    )


def make_figure_includes(run_dir: Path):
    blocks = [_figure_block(path, caption, label) for _, path, caption, label in FIGURES]
    _write(run_dir / "chapter4_figures.tex", "\n".join(blocks))


def _artifact_status(run_dir: Path) -> list[tuple[str, bool]]:
    return [(name, (run_dir / name).exists()) for name in REQUIRED_ARTIFACTS]


def validate_artifacts(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    missing = [name for name, ok in _artifact_status(run_dir) if not ok]
    if missing:
        raise FileNotFoundError("Missing chapter 4 artifacts: " + ", ".join(missing))
    return missing


def make_summary(run_dir: Path):
    selected = {}
    config_path = run_dir / "selected_config.yaml"
    if config_path.exists():
        selected = yaml.safe_load(config_path.read_text(encoding="utf-8")).get("selected", {})

    als_note = None
    external_path = run_dir / "external_baselines.csv"
    if external_path.exists():
        external = pd.read_csv(external_path)
        skipped = external[(external["method"] == "ALS implicit") & (external["status"] == "skipped")]
        if not skipped.empty:
            als_note = f"- ALS implicit: skipped ({skipped.iloc[0].get('reason', '')})."

    lines = [
        "# Chapter 4 Artifacts",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Selected Configuration",
        "",
        f"- epsilon*: `{selected.get('epsilon', {}).get('epsilon')}`",
        f"- k*: `{selected.get('rank', {}).get('nystrom_rank')}`",
        f"- b*: `{selected.get('block', {}).get('block_size')}`",
        als_note,
        "",
        "## Insert Map",
        "",
        "- 4.1: `tables/dataset_table.tex`, `tables/environment_table.tex`",
        "- 4.2: `tables/epsilon_sweep_table.tex`, `figures/epsilon_tradeoff.png`",
        "- 4.3: `tables/rank_sweep_table.tex`, `figures/rank_time_iterations.png`",
        "- 4.4: `tables/block_sweep_table.tex`, `figures/block_time_memory.png`",
        "- 4.5: `tables/internal_baselines_table.tex`, `figures/internal_ease_baselines.png`, `tables/external_baselines_table.tex`, `figures/external_baselines_quality.png`",
        "- Summary table: `tables/selected_parameters_table.tex`",
        "- Ready figure blocks: `chapter4_figures.tex`",
        "",
        "## Artifact Checklist",
        "",
    ]
    for name, ok in _artifact_status(run_dir):
        if name == "chapter4_summary.md":
            ok = True
        lines.append(f"- [{'x' if ok else ' '}] `{name}`")
    _write(run_dir / "chapter4_summary.md", "\n".join(line for line in lines if line is not None) + "\n")


def make_tables(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    make_dataset_table(run_dir, tables_dir)
    make_environment_table(run_dir, tables_dir)
    make_result_tables(run_dir, tables_dir)
    make_selected_parameters_table(run_dir, tables_dir)
    make_figure_includes(run_dir)
    make_summary(run_dir)
    return tables_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    make_tables(args.run_dir)


if __name__ == "__main__":
    main()
