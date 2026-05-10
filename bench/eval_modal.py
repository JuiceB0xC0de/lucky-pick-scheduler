import modal

app = modal.App("deep-chaos-eval")

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

TASKS = "gsm8k,minerva_math,hendrycks_math500,mgsm_direct_en"
WANDB_PROJECT = "deep-chaos-evals"

SECRETS = [
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("WANDB_API_KEY"),
]

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "transformers>=4.51.0",
        "lm_eval[math,vllm]>=0.4.4",
        "wandb",
        "huggingface-hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/vllm": vllm_cache_vol,
}


def _run_eval(model_id: str, tasks: str = TASKS, limit: int = 500, tokenizer: str = ""):
    import subprocess, os
    from huggingface_hub import snapshot_download
    run_name = model_id.split("/")[-1]
    tok = tokenizer or model_id
    # vllm doesn't support subfolder arg — download checkpoint locally first
    parts = model_id.split("/")
    if len(parts) == 3:
        local_path = snapshot_download(
            repo_id="/".join(parts[:2]),
            allow_patterns=f"{parts[2]}/*",
        )
        pretrained = os.path.join(local_path, parts[2])
    else:
        pretrained = model_id
    model_args = (
        f"pretrained={pretrained},"
        f"tokenizer={tok},"
        "dtype=bfloat16,"
        "gpu_memory_utilization=0.85,"
        "max_model_len=8192"
    )
    env = os.environ.copy()
    env["HUGGING_FACE_HUB_TOKEN"] = env.get("HF_TOKEN", "")
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--limit", str(limit),
        "--batch_size", "auto",
        "--gen_kwargs", "max_gen_toks=1024",
        "--output_path", f"/tmp/results/{run_name}",
        "--log_samples",
        "--wandb_args", f"project={WANDB_PROJECT},name={run_name}",
    ]
    subprocess.run(cmd, check=True, env=env)


@app.function(
    image=eval_image,
    gpu="L4",
    timeout=7200,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def eval_3b(model_id: str, tasks: str = TASKS, limit: int = 500):
    _run_eval(model_id, tasks=tasks, limit=limit, tokenizer="Qwen/Qwen2.5-3B-Instruct")


@app.function(
    image=eval_image,
    gpu="H100",
    timeout=7200,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def eval_7b(model_id: str, tasks: str = TASKS, limit: int = 500):
    _run_eval(model_id, tasks=tasks, limit=limit, tokenizer="Qwen/Qwen2.5-7B-Instruct")


@app.function(image=eval_image, gpu="L4", timeout=120, secrets=SECRETS)
def list_tasks():
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    tasks = sorted(tm.all_tasks)
    return "\n".join(tasks)


@app.local_entrypoint()
def show_tasks():
    output = list_tasks.remote()
    keywords = ["math", "gsm", "mgsm", "minerva", "hendrycks", "asdiv", "arith", "drop", "mmlu"]
    matches = [l for l in output.splitlines() if any(k in l.lower() for k in keywords)]
    print("\n".join(matches))


@app.local_entrypoint()
def test():
    """Single model, single task, small limit — verify the pipeline."""
    eval_3b.remote("juiceb0xc0de/benchmark-lucky-pick-19", tasks="gsm8k", limit=50)
    print("Test run complete.")


@app.local_entrypoint()
def run_single(model_id: str, size: str = "3b"):
    """Fire one model eval independently in its own terminal.

    modal run eval_modal.py::run_single --model-id juiceb0xc0de/benchmark-lucky-pick-19
    modal run eval_modal.py::run_single --model-id juiceb0xc0de/benchmark_luckypick_7b_667 --size 7b
    """
    if size == "7b":
        eval_7b.remote(model_id)
    else:
        eval_3b.remote(model_id)


@app.local_entrypoint()
def main():
    handles = (
        [eval_3b.spawn(m) for m in MODELS_3B]
        + [eval_7b.spawn(m) for m in MODELS_7B]
    )
    for h in handles:
        h.get()
    print("All evals complete.")
