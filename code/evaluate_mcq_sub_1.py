"""
Bangla Idiom MCQ Evaluator — Subtask 1 (Select One Meaning)
============================================================
• Supports multiple Ollama models configured in models.yml
• Per-model token/timeout profiles (fixes DeepSeek-R1 CoT truncation)
• Resumes from interruption (per-model checkpoint files)
• --reset MODEL  : delete checkpoint and re-run a specific model
• --only MODEL   : run only specific models, skip others
• Saves raw model outputs and final accuracy results as JSON
• Robust 6-layer response parsing
"""

import json
import re
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import random
import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# ── Load models config from models.yml ───────────────────────────────────────

def _load_models_config() -> dict:
    with open(ROOT / "models.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_models_config()

OLLAMA_BASE_URL = _cfg["ollama"]["base_url"]
MODELS          = _cfg["models"]

MODEL_PROFILES  = {
    name: (profile["num_predict"], profile["timeout"])
    for name, profile in _cfg["model_profiles"].items()
}

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH  = ROOT / "tasks" / "mcq_task_3_subtask_1_select_one_meaning_prompt.json"
RESULT_DIR = ROOT / "results" / "mcq_subtask_1"

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RETRIES   = 3
RETRY_DELAY   = 5
SAVE_EVERY    = 10
TEMPERATURE   = 0.0
VALID_CHOICES = {"1", "2", "3", "4"}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Model profile lookup ──────────────────────────────────────────────────────

def get_model_profile(model: str) -> tuple[int, int]:
    ml = model.lower()
    for pattern, profile in MODEL_PROFILES.items():
        if pattern != "_default" and pattern in ml:
            return profile
    return MODEL_PROFILES["_default"]


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str) -> Optional[str]:
    """
    Extract a single digit choice (1-4) from the model's raw response.

    Strategy (in order):
      1. Strip <think>/<reasoning>/scratchpad blocks (DeepSeek-R1 CoT)
      2. Exact single-digit match on stripped response
      3. Digit alone on its own line
      4. Structured patterns: "Answer: 2", "Option 3", "(4)", etc.
      5. Digit + punctuation at start/end of line
      6. First standalone digit anywhere in text
    """
    if not raw:
        return None

    cleaned = re.sub(
        r"<(?:think|thinking|reasoning|scratchpad)>.*?</(?:think|thinking|reasoning|scratchpad)>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    cleaned = re.sub(r"\*+", "", cleaned).strip()

    if cleaned in VALID_CHOICES:
        return cleaned

    for line in cleaned.splitlines():
        line = line.strip()
        if line in VALID_CHOICES:
            return line

    patterns = [
        r"\bthe\s+(?:correct\s+)?(?:answer|option)\s+is\s+([1-4])\b",
        r"\b(?:answer|option|choice)\s*[:\-]?\s*([1-4])\b",
        r"\(([1-4])\)",
        r"\b([1-4])\s*(?:is\s+(?:the\s+)?correct)\b",
        r"^([1-4])[.\):\s]",
        r"([1-4])[.\):\s]?$",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1) in VALID_CHOICES:
            return m.group(1)

    m = re.search(r"\b([1-4])\b", cleaned)
    if m and m.group(1) in VALID_CHOICES:
        return m.group(1)

    return None


# ── Ollama API ────────────────────────────────────────────────────────────────

def query_ollama(model: str, prompt: str) -> tuple[str, float]:
    num_predict, timeout = get_model_profile(model)
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": num_predict,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0      = time.time()
            resp    = requests.post(url, json=payload, timeout=timeout)
            latency = time.time() - t0
            resp.raise_for_status()
            data    = resp.json()
            raw     = data.get("response", "")

            if data.get("done_reason") == "length" and not parse_response(raw):
                log.debug(
                    "Model '%s' hit num_predict=%d cap without parseable answer.",
                    model, num_predict,
                )
            return raw, latency

        except requests.exceptions.Timeout:
            log.warning("Timeout (>%ds) on attempt %d/%d for '%s'",
                        timeout, attempt, MAX_RETRIES, model)
        except requests.exceptions.ConnectionError:
            log.warning("Connection error on attempt %d/%d. Is Ollama running?",
                        attempt, MAX_RETRIES)
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP error %s on attempt %d/%d", e, attempt, MAX_RETRIES)
        except Exception as e:
            log.warning("Unexpected error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return "", 0.0


def check_ollama_connection() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ── Checkpoint / file helpers ─────────────────────────────────────────────────

def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def checkpoint_path(result_dir: Path, model: str, sample_tag: str = "") -> Path:
    return result_dir / f"{safe_name(model)}{sample_tag}_checkpoint.json"


def output_path(result_dir: Path, model: str, sample_tag: str = "") -> Path:
    return result_dir / f"{safe_name(model)}{sample_tag}_outputs.json"


def results_summary_path(result_dir: Path) -> Path:
    return result_dir / "results_summary.json"


def _atomic_write(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("  Resuming from checkpoint: %s  (%d items done)", path.name, len(data))
            return data
        except Exception as e:
            log.warning("  Corrupted checkpoint (%s) — starting fresh.", e)
    return {}


def load_results_summary(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(model: str, dataset: list[dict], result_dir: Path, sample_tag: str = "") -> dict:
    log.info("\n%s", "=" * 60)
    log.info("  Model  : %s", model)
    num_predict, timeout = get_model_profile(model)
    log.info("  Profile: num_predict=%d  timeout=%ds", num_predict, timeout)
    log.info("  Items  : %d", len(dataset))
    log.info("%s", "=" * 60)

    ckpt_path   = checkpoint_path(result_dir, model, sample_tag)
    records     = load_checkpoint(ckpt_path)

    correct     = sum(1 for r in records.values() if r.get("correct"))
    unparseable = sum(1 for r in records.values() if r.get("parsed") is None)
    total_done  = len(records)

    pbar = tqdm(total=len(dataset), initial=total_done, desc=model, unit="item")

    for item in dataset:
        pid = item["prompt_id"]
        if pid in records:
            continue

        raw, latency = query_ollama(model, item["prompt"])
        parsed       = parse_response(raw)
        is_correct   = parsed == item["ground_truth"]

        records[pid] = {
            "prompt_id"   : pid,
            "prompt"      : item["prompt"],
            "ground_truth": item["ground_truth"],
            "raw_response": raw,
            "parsed"      : parsed,
            "correct"     : is_correct,
            "latency_s"   : round(latency, 3),
        }

        if is_correct:
            correct += 1
        if parsed is None:
            unparseable += 1
        total_done += 1

        pbar.update(1)
        pbar.set_postfix(acc=f"{correct/total_done:.1%}", unparse=unparseable)

        if total_done % SAVE_EVERY == 0:
            _atomic_write(ckpt_path, records)

    pbar.close()

    _atomic_write(ckpt_path, records)
    _atomic_write(output_path(result_dir, model, sample_tag), records)

    n        = len(records)
    accuracy = correct / n if n > 0 else 0.0

    result = {
        "model"       : model,
        "total"       : n,
        "correct"     : correct,
        "incorrect"   : n - correct - unparseable,
        "unparseable" : unparseable,
        "accuracy"    : round(accuracy, 6),
        "accuracy_pct": round(accuracy * 100, 2),
        "sample_tag"  : sample_tag if sample_tag else "full",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    log.info(
        "\n  ✓ %s: %d/%d correct  →  Accuracy = %.2f%%  (unparseable: %d)",
        model, correct, n, accuracy * 100, unparseable,
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bangla Idiom MCQ Evaluator — Subtask 1",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Model tags to evaluate (default: from models.yml)")
    parser.add_argument("--data", default=DATA_PATH,
                        help="Path to MCQ JSON dataset")
    parser.add_argument("--result-dir", default=RESULT_DIR,
                        help="Output directory")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    parser.add_argument("--only", nargs="+", metavar="MODEL",
                        help="Run only these model(s), skip all others")
    parser.add_argument("--reset", nargs="+", metavar="MODEL",
                        help="Delete checkpoint for these model(s) before running")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Evaluate on a random subset of N items")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample (default: 42)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(result_dir / "eval.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    if not check_ollama_connection():
        log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_BASE_URL)
        sys.exit(1)

    if args.list_models:
        for m in list_ollama_models():
            print(f"  {m}")
        sys.exit(0)

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Dataset not found: %s", data_path)
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    log.info("Loaded %d items from %s", len(dataset), data_path.name)

    sample_tag = ""
    if args.sample:
        if args.sample >= len(dataset):
            log.warning("--sample %d >= dataset size %d; using full dataset.",
                        args.sample, len(dataset))
        else:
            random.seed(args.seed)
            dataset    = random.sample(dataset, args.sample)
            sample_tag = f"_sample{args.sample}_seed{args.seed}"
            log.info("Sampled %d items (seed=%d)  →  tag: '%s'",
                     args.sample, args.seed, sample_tag)

    models_to_run    = args.only if args.only else args.models
    available_models = set(list_ollama_models())

    for m in models_to_run:
        if m not in available_models:
            log.warning("Model '%s' not found in Ollama (run `ollama pull %s` first)", m, m)

    summary_path = results_summary_path(result_dir)
    summary      = load_results_summary(summary_path)

    if args.reset:
        for m in args.reset:
            ckpt = checkpoint_path(result_dir, m)
            if ckpt.exists():
                ckpt.unlink()
                log.info("Reset: deleted checkpoint for '%s'", m)
            if m in summary:
                del summary[m]
                _atomic_write(summary_path, summary)

    log.info("\nModel profiles:")
    for m in models_to_run:
        np, to = get_model_profile(m)
        log.info("  %-30s  num_predict=%-6d  timeout=%ds", m, np, to)

    for model in models_to_run:
        result = evaluate_model(model, dataset, result_dir, sample_tag=sample_tag)
        summary[model] = result
        _atomic_write(summary_path, summary)

    log.info("\n%s", "=" * 60)
    log.info("LEADERBOARD")
    log.info("=" * 60)
    for rank, r in enumerate(
        sorted(summary.values(), key=lambda x: x["accuracy"], reverse=True), 1
    ):
        log.info(
            "  #%-2d  %-28s  Acc=%6.2f%%  (%d/%d)  unparseable=%d",
            rank, r["model"], r["accuracy_pct"],
            r["correct"], r["total"], r["unparseable"],
        )
    log.info("=" * 60)
    log.info("Results saved to: %s", result_dir)


if __name__ == "__main__":
    main()
