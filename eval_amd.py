"""Native AMD MI300X / ROCm eval driver for the DeepChaos benchmark suite.

Runs lm-eval-harness with the vLLM backend against the trained checkpoints
on a local MI300X.  No Modal, no Docker — just `python eval_amd.py`.

Prerequisites (one-time, on the droplet):
    bash setup_amd.sh                                  # installs the trainer stack
    pip install "lm_eval[math,vllm]>=0.4.4"            # eval harness
    pip install --pre vllm \\
        --index-url https://download.pytorch.org/whl/rocm6.2  # ROCm-built vllm

(The `pip install vllm` call must hit a ROCm wheel, not a CUDA one.  vLLM's
MI300X path is upstream from 0.6+.  Confirm with `python -c "import vllm; print(vllm.__version__)"`.)

Usage:
    # Run the full sweep — 5 × 3B + 5 × 7B sequentially on one MI300X.
    python eval_amd.py --all

    # Single model.
    python eval_amd.py --model juiceb0xc0de/benchmark-lucky-pick-19

    # Subset of tasks, smaller limit (smoke test the pipeline).
    python eval_amd.py --model juiceb0xc0de/benchmark-lucky-pick-19 \\
        --tasks gsm8k --limit 50

    # Just the 3B sweep.
    python eval_amd.py --3b

Notes:
    * One MI300X = 192GB VRAM, so 3B and 7B both fit with room to spare.
    * Models are evaluated sequentially.  Wall-clock for the full sweep is
      ~6-8h depending on task config.
    * Results land in ./results/<run_name>/ and (if WANDB_API_KEY is set)
      stream to W&B project "deep-chaos-evals".
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ROCm-friendly defaults — same env as the trainer scripts in this repo.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


MODELS_3B = [
    "juiceb0xc0de/benchmark-lucky-pick-3047",
    "juiceb0xc0de/benchmark-lucky-pick-666",
    "juiceb0xc0de/benchmark-lucky-pick-19",
    "juiceb0xc0de/lucky-pick-baseline",
    "Qwen/Qwen2.5-3B-Instruct",
]

MODELS_7B = [
    "juiceb0xc0de/benchmark-lucky-pick-7b-14",
    "juiceb0xc0de/benchmark-luckypick-7b-667",
    "juiceb0xc0de/benchmark-luckypick-7b-555",
    "juiceb0xc0de/benchmark-luckypick-7b-19",
    "juiceb0xc0de/benchmark-fft-7b-19",
]

DEFAULT_TASKS = "gsm8k,minerva_math,hendrycks_math500,mgsm_direct_en"
WANDB_PROJECT = "deep-chaos-evals"
RESULTS_DIR = Path("./results")


def _resolve_pretrained(model_id: str) -> str:
    """Some Hub repos store the actual checkpoint under a subfolder.
    vLLM doesn't accept a subfolder arg, so snapshot-download into a
    local path and point vLLM at the subfolder directly."""
    parts = model_id.split("/")
    if len(parts) != 3:
        return model_id
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        repo_id="/".join(parts[:2]),
        allow_patterns=f"{parts[2]}/*",
    )
    return os.path.join(local_path, parts[2])


def _tokenizer_for(model_id: str) -> str:
    """Match a fine-tuned checkpoint to its base tokenizer (DeepChaos
    runs may have stripped/altered the tokenizer artifacts)."""
    if "7b" in model_id.lower():
        return "Qwen/Qwen2.5-7B-Instruct"
    return "Qwen/Qwen2.5-3B-Instruct"


def run_eval(model_id: str, tasks: str, limit: int) -> int:
    run_name = model_id.replace("/", "__")
    pretrained = _resolve_pretrained(model_id)
    tokenizer = _tokenizer_for(model_id)

    model_args = (
        f"pretrained={pretrained},"
        f"tokenizer={tokenizer},"
        "dtype=bfloat16,"
        "gpu_memory_utilization=0.85,"
        "max_model_len=8192"
    )

    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--limit", str(limit),
        "--batch_size", "auto",
        "--gen_kwargs", "max_gen_toks=1024",
        "--output_path", str(output_dir),
        "--log_samples",
    ]
    if os.environ.get("WANDB_API_KEY"):
        cmd += ["--wandb_args", f"project={WANDB_PROJECT},name={run_name}"]

    env = os.environ.copy()
    env.setdefault("HUGGING_FACE_HUB_TOKEN", env.get("HF_TOKEN", ""))

    print(f"\n{'=' * 70}\n▶ Evaluating {model_id}\n{'=' * 70}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep = p.add_mutually_exclusive_group()
    sweep.add_argument("--all", action="store_true", help="run the full 3B + 7B sweep")
    sweep.add_argument("--3b", dest="only_3b", action="store_true", help="run the 3B sweep only")
    sweep.add_argument("--7b", dest="only_7b", action="store_true", help="run the 7B sweep only")
    sweep.add_argument("--model", help="single model id (Hub repo or repo/subfolder)")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help=f"lm-eval task list (default: {DEFAULT_TASKS})")
    p.add_argument("--limit", type=int, default=500, help="examples per task (default: 500)")
    args = p.parse_args()

    if args.model:
        targets = [args.model]
    elif args.only_3b:
        targets = MODELS_3B
    elif args.only_7b:
        targets = MODELS_7B
    elif args.all:
        targets = MODELS_3B + MODELS_7B
    else:
        p.error("pick one of --all / --3b / --7b / --model")

    failures = []
    for model_id in targets:
        rc = run_eval(model_id, tasks=args.tasks, limit=args.limit)
        if rc != 0:
            print(f"!! {model_id} returned non-zero ({rc})")
            failures.append(model_id)

    print(f"\nFinished: {len(targets) - len(failures)}/{len(targets)} succeeded.")
    if failures:
        print("Failed:")
        for m in failures:
            print(f"  - {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
