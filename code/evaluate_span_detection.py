#!/usr/bin/env python3
"""
Task 2: Bangla Idiom Span Detection Evaluator
Models via Ollama | Metrics: % Unigram Overlap (F1), Levenshtein Distance, BERTScore
Supports: 0-shot, 3-shot, 5-shot | Resume-safe checkpointing | --sample N
Models and profiles loaded from models.yml
"""

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# ── Load models config from models.yml ───────────────────────────────────────

def _load_models_config() -> dict:
    with open(ROOT / "models.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_models_config()

OLLAMA_URL      = _cfg["ollama"]["base_url"] + "/api/generate"
MODELS          = _cfg["models"]
THINKING_MODELS = set(_cfg["thinking_models"])

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH  = ROOT / "tasks" / "task_2_span_detection_prompt.json"
RESULT_DIR = ROOT / "results" / "span_detection"

# ── Constants ─────────────────────────────────────────────────────────────────

SHOT_TYPES     = ["0-shot", "3-shot", "5-shot"]
SHOT_KEYS      = {
    "0-shot": "0-shot-prompt",
    "3-shot": "3-shot-prompt",
    "5-shot": "5-shot-prompt",
}
OLLAMA_TIMEOUT = 180
OLLAMA_RETRIES = 3

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_generate(model: str, prompt: str) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            log.warning(f"  [attempt {attempt}/{OLLAMA_RETRIES}] {model}: {e}")
            if attempt < OLLAMA_RETRIES:
                time.sleep(3 * attempt)
    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_QUOTE_RE = re.compile(r"["“”‘’']")
_PREFIX_RE = re.compile(
    r"(?i)^(detected[\s_]idiom\s*[-:–]?\s*"
    r"|idiom\s*[-:–]?\s*"
    r"|answer\s*[-:–]?\s*"
    r"|output\s*[-:–]?\s*"
    r"|span\s*[-:–]?\s*)"
)


def parse_response(raw: str, model: str) -> str:
    if not raw:
        return ""
    text = raw
    if model in THINKING_MODELS or _THINK_RE.search(text):
        text = _THINK_RE.sub("", text)
    text = re.sub(r"[*_`]+", "", text)
    text = _QUOTE_RE.sub("", text)
    text = _PREFIX_RE.sub("", text.strip())
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def unigram_overlap(pred: str, gold: str) -> dict:
    pred_tokens = pred.strip().split()
    gold_tokens = gold.strip().split()

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": True}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": False}

    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = sum((pred_c & gold_c).values())

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact = pred.strip() == gold.strip()

    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact}


def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def normalised_ld(pred: str, gold: str) -> float:
    d = levenshtein(pred, gold)
    return d / max(len(pred), len(gold), 1)


def compute_bertscore(preds: list, refs: list, lang: str = "bn") -> dict:
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(
            preds, refs,
            lang=lang,
            model_type="bert-base-multilingual-cased",
            verbose=False,
            batch_size=64,
        )
        return {
            "precision": P.tolist(),
            "recall":    R.tolist(),
            "f1":        F.tolist(),
            "mean_precision": round(100 * float(P.mean()), 4),
            "mean_recall":    round(100 * float(R.mean()), 4),
            "mean_f1":        round(100 * float(F.mean()), 4),
        }
    except ImportError:
        log.warning("bert_score not installed — skipping. pip install bert-score")
        return {}
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        return {}


def build_corpus_metrics(records: list, bs: dict) -> dict:
    preds = [r["prediction"]   for r in records]
    refs  = [r["ground_truth"] for r in records]
    n     = len(records)

    uni_scores    = [unigram_overlap(p, g) for p, g in zip(preds, refs)]
    ld_raw        = [levenshtein(p, g)     for p, g in zip(preds, refs)]
    nld           = [normalised_ld(p, g)   for p, g in zip(preds, refs)]
    exact_matches = sum(1 for s in uni_scores if s["exact_match"])

    corpus = {
        "unigram_overlap_pct":   round(100 * sum(s["f1"]        for s in uni_scores) / n, 4),
        "levenshtein_mean":      round(sum(ld_raw) / n, 4),
        "unigram_precision_pct": round(100 * sum(s["precision"] for s in uni_scores) / n, 4),
        "unigram_recall_pct":    round(100 * sum(s["recall"]    for s in uni_scores) / n, 4),
        "exact_match_pct":       round(100 * exact_matches / n, 4),
        "exact_match_count":     exact_matches,
        "normalised_ld_mean":    round(sum(nld) / n, 4),
        "total_samples":         n,
    }

    if bs:
        corpus["bertscore_precision_pct"] = bs["mean_precision"]
        corpus["bertscore_recall_pct"]    = bs["mean_recall"]
        corpus["bertscore_f1_pct"]        = bs["mean_f1"]

    for i, r in enumerate(records):
        r["metrics"] = {
            "unigram_overlap_pct": round(100 * uni_scores[i]["f1"], 4),
            "unigram_precision":   round(100 * uni_scores[i]["precision"], 4),
            "unigram_recall":      round(100 * uni_scores[i]["recall"], 4),
            "exact_match":         uni_scores[i]["exact_match"],
            "levenshtein":         ld_raw[i],
            "normalised_ld":       round(nld[i], 4),
        }
        if bs:
            r["metrics"]["bertscore_f1"] = round(100 * bs["f1"][i], 4)

    return corpus


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")

def output_path(model: str, shot: str) -> Path:
    return RESULT_DIR / f"{_safe(model)}_{shot}_outputs.json"

def result_path(model: str, shot: str) -> Path:
    return RESULT_DIR / f"{_safe(model)}_{shot}_results.json"

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return {item["prompt_id"]: item for item in items}
    return {}

def save_outputs(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── Evaluation core ───────────────────────────────────────────────────────────

def evaluate_model_shot(model: str, shot: str, dataset: list,
                        sample: int | None, force: bool) -> dict:
    shot_key = SHOT_KEYS[shot]
    out_path = output_path(model, shot)
    res_path = result_path(model, shot)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    existing = {} if force else load_checkpoint(out_path)
    data     = dataset[:sample] if sample else dataset

    log.info("=" * 60)
    log.info(f"  Model : {model}  |  Shot : {shot}")
    log.info(f"  Items : {len(data)}"
             + (f"  (resuming from {len(existing)})" if existing else ""))
    log.info("=" * 60)

    records  = list(existing.values())
    done_ids = set(existing.keys())
    pending  = [item for item in data if item["prompt_id"] not in done_ids]

    if pending:
        pbar = tqdm(pending, desc=f"{model} [{shot}]", unit="item")
        for item in pbar:
            pid    = item["prompt_id"]
            prompt = item[shot_key]
            gt     = item["ground_truth"]

            raw  = ollama_generate(model, prompt)
            pred = parse_response(raw, model)

            records.append({
                "prompt_id":    pid,
                "ground_truth": gt,
                "raw_response": raw,
                "prediction":   pred,
            })
            done_ids.add(pid)
            save_outputs(out_path, records)
        pbar.close()

    log.info(f"Computing metrics for {model} [{shot}] …")
    preds = [r["prediction"]   for r in records]
    refs  = [r["ground_truth"] for r in records]
    bs    = compute_bertscore(preds, refs)

    corpus = build_corpus_metrics(records, bs)
    save_outputs(out_path, records)

    result_payload = {"model": model, "shot": shot, "corpus_metrics": corpus}
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    has_bs = bool(bs)
    log.info(
        f"  ✓ {model} [{shot}]  "
        f"Unigram%={corpus['unigram_overlap_pct']:.2f}  "
        f"LD={corpus['levenshtein_mean']:.2f}  "
        f"EM={corpus['exact_match_pct']:.2f}%  "
        f"NLD={corpus['normalised_ld_mean']:.4f}"
        + (f"  BERTf1={corpus['bertscore_f1_pct']:.2f}%" if has_bs else "")
    )
    return result_payload


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    has_bs = any("bertscore_f1_pct" in r["corpus_metrics"] for r in all_results)

    log.info("\n" + "=" * 80)
    log.info("SUMMARY — Task 2: Span Detection")
    log.info("=" * 80)

    hdr = (f"{'#':<3} {'Model':<24} {'Fewshot':<8} "
           f"{'% Unigram Overlap':>18} {'LD':>6}")
    if has_bs:
        hdr += f"  {'BERTScore F1%':>13}"
    hdr += f"  {'EM%':>6}"
    log.info(hdr)
    log.info("-" * 80)

    for idx, r in enumerate(all_results, 1):
        c   = r["corpus_metrics"]
        row = (f"{idx:<3} {r['model']:<24} {r['shot']:<8} "
               f"{c['unigram_overlap_pct']:>18.2f} "
               f"{c['levenshtein_mean']:>6.2f}")
        if has_bs:
            row += f"  {c.get('bertscore_f1_pct', float('nan')):>13.2f}"
        row += f"  {c['exact_match_pct']:>6.2f}"
        log.info(row)

    log.info("=" * 80)
    log.info("Columns: % Unigram Overlap = word-level F1×100  |  LD = mean Levenshtein distance")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bangla Idiom Span Detection Evaluator")
    parser.add_argument("--models",      nargs="+", default=MODELS)
    parser.add_argument("--shots",       nargs="+", default=SHOT_TYPES, choices=SHOT_TYPES)
    parser.add_argument("--sample",      type=int,  default=None,
                        help="Evaluate only first N samples")
    parser.add_argument("--data",        default=DATA_PATH)
    parser.add_argument("--force",       action="store_true",
                        help="Ignore checkpoints and re-run everything")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip inference; recompute metrics from saved outputs")
    args = parser.parse_args()

    log.info(f"Loading data from {args.data}")
    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    log.info(f"Loaded {len(dataset)} items")

    all_results = []

    for model in args.models:
        for shot in args.shots:
            if args.metrics_only:
                out_path = output_path(model, shot)
                if not out_path.exists():
                    log.warning(f"No output file for {model} [{shot}] — skipping")
                    continue
                with open(out_path, encoding="utf-8") as f:
                    records = json.load(f)
                preds  = [r["prediction"]   for r in records]
                refs   = [r["ground_truth"] for r in records]
                bs     = compute_bertscore(preds, refs)
                corpus = build_corpus_metrics(records, bs)
                save_outputs(out_path, records)
                result_payload = {"model": model, "shot": shot, "corpus_metrics": corpus}
                res_path = result_path(model, shot)
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(result_payload, f, ensure_ascii=False, indent=2)
                all_results.append(result_payload)
            else:
                try:
                    r = evaluate_model_shot(
                        model, shot, dataset,
                        sample=args.sample, force=args.force,
                    )
                    all_results.append(r)
                except KeyboardInterrupt:
                    log.warning("Interrupted — progress saved. Re-run to continue.")
                    raise
                except Exception as e:
                    log.error(f"Failed {model} [{shot}]: {e}", exc_info=True)

    if all_results:
        print_summary(all_results)
        summary_path = RESULT_DIR / "summary_all.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        log.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
