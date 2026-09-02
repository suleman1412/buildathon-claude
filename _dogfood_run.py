"""One-off driver for validating our own team's GNN-Path-Planning repo against
its presentation PDF. Same as validate.py's run_pipeline, except: (1) the repo
URL is supplied directly since the slide deck doesn't link it in text, and (2)
a downsized reconstruction of the notebook's train/eval logic is dropped into
the clone right after cloning, since this repo's src/train.py is an empty
stub -- the real logic only exists in a Colab notebook with GPU-only installs
and IPython magics that won't run as a plain script. Everything after that
(install deps, LLM proposes how to run it, debug-loop retries, verify,
report, dashboard) is the real pipeline, unmodified. Not committed -- local
scratch driver only, and never pushes to the teammate's actual repo."""
import shutil
import sys
from pathlib import Path

from src.config import Config
from src.dashboard import record_run
from src.debug_loop import run_with_debug_loop
from src.execution import propose_run_plan, truncate_log
from src.extraction import extract_claim
from src.llm_client import AnthropicLLMClient, SpendCapExceeded
from src.pdf_extract import extract_text
from src.provisioning import clone_repo, install_requirements
from src.verification import verify

PDF = "gnn_presentation_researchgate.pdf"
REPO_URL = "https://github.com/shaizalyasin/GNN-Path-Planning.git"
DOWNSIZED_SCRIPT = Path("_gnn_downsized_script.py").read_text()

llm = AnthropicLLMClient(max_spend_usd=1.0)
config = Config(subprocess_timeout_sec=180, max_runtime_retries=3)

try:
    paper_text = extract_text(PDF)
    claim = extract_claim(llm, paper_text)
    claim.github_repo = REPO_URL

    workdir = Path(config.workdir) / "gnn_path_planning"
    repo_dir = workdir / "repo"
    shutil.rmtree(workdir, ignore_errors=True)
    clone_repo(claim.github_repo, repo_dir)
    (repo_dir / "repro_downsized.py").write_text(DOWNSIZED_SCRIPT)
    install_requirements(repo_dir)

    plan = propose_run_plan(llm, repo_dir, claim.dataset, claim.claimed_metric_name)
    print(f"Proposed command: {plan.command}")
    print(f"Proposed metric_regex: {plan.metric_regex}")

    result, final_command, final_metric_regex, reproduced_value = run_with_debug_loop(
        llm, plan.command, plan.metric_regex, repo_dir,
        max_retries=config.max_runtime_retries,
        truncate_lines=config.log_truncate_lines,
        timeout_sec=config.subprocess_timeout_sec,
        claimed_metric_name=claim.claimed_metric_name,
    )
    print(f"Final command: {final_command}")
    print(f"Final metric_regex: {final_metric_regex}")

    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}). Command: {final_command}")
        print("--- stderr tail ---")
        print(truncate_log(result.stderr, 30))
        print("--- stdout tail ---")
        print(truncate_log(result.stdout, 30))
        sys.exit(1)

    if reproduced_value is None:
        print(f"No metric matched. Command: {final_command}  regex: {final_metric_regex}")
        print("--- stdout tail ---")
        print(truncate_log(result.stdout, 40))
        sys.exit(1)

    verification = verify(claim, reproduced_value, config.metric_tolerance_relative)
except SpendCapExceeded as e:
    print(f"STOPPED: {e}")
    sys.exit(2)
finally:
    print(f"Estimated spend this run: ${llm.spend_usd:.4f}")

report = verification.to_markdown()
with open("report.md", "w", encoding="utf-8") as f:
    f.write(report)
print(report.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))

record_run(
    verification,
    backend="anthropic",
    model=getattr(llm, "_model", None),
    spend_usd=llm.spend_usd,
    pdf_path=PDF,
    dashboard_data_path=Path("dashboard/data/runs.json"),
)
