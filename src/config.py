from dataclasses import dataclass


@dataclass
class Config:
    max_runtime_retries: int = 5
    log_truncate_lines: int = 30
    subprocess_timeout_sec: int = 900
    metric_tolerance_relative: float = 0.05
    workdir: str = "./runs"
    hf_eval_max_examples: int = 200  # cap eval-set size for speed; enough for a stable accuracy estimate
