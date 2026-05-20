import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_default_config(cfg: dict) -> None:
    for section in ("data", "model", "solver", "runtime"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    model = cfg["model"]
    solver = cfg["solver"]
    runtime = cfg["runtime"]

    if model["reg"] <= 0:
        raise ValueError("model.reg must be positive")
    if model["max_iter"] <= 0:
        raise ValueError("model.max_iter must be positive")
    if model["k_recommend"] <= 0:
        raise ValueError("model.k_recommend must be positive")
    if model.get("top_l") is not None and model["top_l"] <= 0:
        raise ValueError("model.top_l must be positive or null")

    if solver["epsilon"] <= 0:
        raise ValueError("solver.epsilon must be positive")
    if solver["block_size"] <= 0:
        raise ValueError("solver.block_size must be positive")
    if solver["nystrom_rank"] < 0:
        raise ValueError("solver.nystrom_rank must be non-negative")
    if solver.get("nystrom_oversample", 0) < 0:
        raise ValueError("solver.nystrom_oversample must be non-negative")

    if runtime["user_batch_size"] <= 0:
        raise ValueError("runtime.user_batch_size must be positive")
    if runtime.get("item_batch_size") is not None and runtime["item_batch_size"] <= 0:
        raise ValueError("runtime.item_batch_size must be positive or null")
    if runtime["max_memory_gb"] <= 0:
        raise ValueError("runtime.max_memory_gb must be positive")

    selection = cfg.get("selection", {})
    if selection:
        if selection.get("tau_recall", 0.0) < 0:
            raise ValueError("selection.tau_recall must be non-negative")
        if selection.get("tau_ndcg", 0.0) < 0:
            raise ValueError("selection.tau_ndcg must be non-negative")
        q_min = selection.get("q_min", 0.0)
        if not 0.0 <= q_min <= 1.0:
            raise ValueError("selection.q_min must be in [0, 1]")
