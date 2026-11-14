#!/usr/bin/env python3
"""
Task 1: Bangla Idiom Paraphrase Quality Evaluator
==================================================
Workflow:
  Source sentence → model paraphrase
  Target          = sentence with idiom replaced by each meaning
  Metrics across all (paraphrase, best-target) pairs

Metrics
-------
  ROUGE-1/2/3/L   manual implementation (no library)
  BERTScore P/R/F1
  Cosine Similarity:
    LaBSE  (sentence-transformers/LaBSE)         ← local, always computed
    BGE-M3 (BAAI/bge-m3, stored as 'laser' key)  ← local, replaces fairseq LASER
    OpenAI (text-embedding-3-large)              ← paid, set USE_PAID_EMBEDDINGS=True
    Gemini (gemini-embedding-001)                ← paid, set USE_PAID_EMBEDDINGS=True

Models and profiles loaded from models.yml
API keys loaded from config.yml
"""

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
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

# ── Load API keys from config.yml ─────────────────────────────────────────────

def _load_api_keys() -> dict:
    cfg_path = ROOT / "config.yml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("api_keys", {})

_api_keys = _load_api_keys()

# Toggle: set True after filling in API keys in config.yml
USE_PAID_EMBEDDINGS = False
OPENAI_API_KEY      = _api_keys.get("openai", "")
GEMINI_API_KEY      = _api_keys.get("gemini", "")

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH   = ROOT / "data"   / "Bagdhara_v3_merged.json"
PROMPT_PATH = ROOT / "tasks"  / "task_1_paraphrase_quality_prompt.json"
RESULT_DIR  = ROOT / "results" / "paraphrase_quality"

# ── Constants ─────────────────────────────────────────────────────────────────

SHOT_TYPES     = ["0-shot", "3-shot", "5-shot"]
SHOT_KEYS      = {
    "0-shot": "0-shot-prompt",
    "3-shot": "3-shot-prompt",
    "5-shot": "5-shot-prompt",
}
OLLAMA_TIMEOUT = 360
OLLAMA_RETRIES = 3

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Dependency check ──────────────────────────────────────────────────────────

def _check_dependencies():
    missing = []
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers  →  pip install sentence-transformers")
    try:
        import bert_score             # noqa: F401
    except ImportError:
        missing.append("bert_score            →  pip install bert-score")

    if missing:
        log.error(
            "\n\nMISSING PACKAGES — cosine / BERTScore results will be EMPTY\n"
            + "\n".join(f"  ✗  {m}" for m in missing)
            + "\n\nInstall and re-run.\n"
        )
        raise SystemExit(1)
    log.info("✓ sentence-transformers and bert_score present")

_check_dependencies()


# ── Data preparation ──────────────────────────────────────────────────────────

def build_dataset(bagdhara_path: str, prompt_path: str) -> list:
    with open(bagdhara_path, encoding="utf-8") as f:
        bagdhara = json.load(f)
    with open(prompt_path, encoding="utf-8") as f:
        prompts = json.load(f)

    lookup  = {(x.get("sentence") or "").strip(): x for x in bagdhara}
    _src_re = re.compile(r'"(.+?)"\.\s*\n', re.DOTALL)

    def extract_source(text: str) -> str:
        m = _src_re.search(text)
        return m.group(1).strip() if m else ""

    dataset = []
    for idx, p in enumerate(prompts):
        src  = extract_source(p["0-shot-prompt"])
        item = lookup.get(src)
        if not item:
            continue

        idiom    = (item.get("idiom")    or "").strip()
        sentence = (item.get("sentence") or "").strip()
        meanings = item.get("meaning")   or []

        if not idiom or idiom not in sentence:
            continue

        targets = [sentence.replace(idiom, " " + m + " ") for m in meanings]

        dataset.append({
            "idx":           idx,
            "source":        sentence,
            "idiom":         idiom,
            "meanings":      meanings,
            "targets":       targets,
            "0-shot-prompt": p["0-shot-prompt"],
            "3-shot-prompt": p["3-shot-prompt"],
            "5-shot-prompt": p["5-shot-prompt"],
        })

    log.info(
        f"Dataset: {len(dataset)} items  "
        f"(total targets: {sum(len(d['targets']) for d in dataset)})"
    )
    return dataset


# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_generate(model: str, prompt: str) -> str | None:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
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


# ── Response parsing ──────────────────────────────────────────────────────────

_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_QUOTE_RE  = re.compile(r'["""'\'']')
_PREFIX_RE = re.compile(
    r"(?i)^(paraphrase\s*[-:–]?\s*"
    r"|output\s*[-:–]?\s*"
    r"|answer\s*[-:–]?\s*"
    r"|result\s*[-:–]?\s*)"
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


# ── Manual ROUGE ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return text.strip().split()

def get_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def rouge_n(pred_tokens: list, ref_tokens: list, n: int) -> dict:
    pred_ng = get_ngrams(pred_tokens, n)
    ref_ng  = get_ngrams(ref_tokens,  n)

    if not pred_ng and not ref_ng:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_ng or not ref_ng:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common    = sum((pred_ng & ref_ng).values())
    precision = common / sum(pred_ng.values())
    recall    = common / sum(ref_ng.values())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def lcs_length(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]

def rouge_l(pred_tokens: list, ref_tokens: list) -> dict:
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs  = lcs_length(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec  = lcs / len(ref_tokens)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def compute_rouge_pair(pred: str, ref: str) -> dict:
    p = tokenize(pred)
    r = tokenize(ref)
    return {
        "rouge1": rouge_n(p, r, 1),
        "rouge2": rouge_n(p, r, 2),
        "rouge3": rouge_n(p, r, 3),
        "rougeL": rouge_l(p, r),
    }

def best_rouge(pred: str, targets: list) -> dict:
    best = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for k in ("rouge1", "rouge2", "rouge3", "rougeL")}
    for tgt in targets:
        scores = compute_rouge_pair(pred, tgt)
        for metric, vals in scores.items():
            if vals["f1"] > best[metric]["f1"]:
                best[metric] = vals
    return best


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(preds: list, refs: list) -> dict:
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(
            preds, refs,
            lang="bn",
            model_type="bert-base-multilingual-cased",
            verbose=False,
            batch_size=64,
        )
        return {
            "per_sample_f1":  [round(v * 100, 4) for v in F.tolist()],
            "mean_precision": round(100 * float(P.mean()), 4),
            "mean_recall":    round(100 * float(R.mean()), 4),
            "mean_f1":        round(100 * float(F.mean()), 4),
        }
    except ImportError:
        log.warning("bert_score not installed — skip. pip install bert-score")
        return {}
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        return {}


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _mean_cosine(emb_preds: np.ndarray, emb_refs: np.ndarray) -> dict:
    scores = [_cosine(emb_preds[i], emb_refs[i]) for i in range(len(emb_preds))]
    return {
        "per_sample": [round(s, 6) for s in scores],
        "mean":       round(float(np.mean(scores)), 6),
    }

_labse_model = None
def _get_labse():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading LaBSE model …")
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    return _labse_model

def compute_labse(preds: list, refs: list) -> dict:
    try:
        model = _get_labse()
        ep = model.encode(preds, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
        er = model.encode(refs,  batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
        scores = [float(np.dot(ep[i], er[i])) for i in range(len(ep))]
        return {"per_sample": [round(s, 6) for s in scores],
                "mean":       round(float(np.mean(scores)), 6)}
    except Exception as e:
        log.error(f"LaBSE FAILED: {e}", exc_info=True)
        return {}

_bgem3_model = None
def _get_bgem3():
    global _bgem3_model
    if _bgem3_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading BAAI/bge-m3 (LASER replacement, ~2.2 GB on first run) …")
        _bgem3_model = SentenceTransformer("BAAI/bge-m3")
    return _bgem3_model

def compute_laser(preds: list, refs: list) -> dict:
    """BGE-M3 cosine — stored under 'laser' key for backward compatibility."""
    try:
        model = _get_bgem3()
        ep = model.encode(preds, batch_size=32, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
        er = model.encode(refs,  batch_size=32, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)
        scores = [float(np.dot(ep[i], er[i])) for i in range(len(ep))]
        return {"per_sample": [round(s, 6) for s in scores],
                "mean":       round(float(np.mean(scores)), 6)}
    except Exception as e:
        log.error(f"BGE-M3 FAILED: {e}", exc_info=True)
        return {}

def compute_openai_embedding(preds: list, refs: list) -> dict:
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        def _embed_batch(texts):
            resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
            return np.array([r.embedding for r in resp.data])
        BATCH = 128
        ep = np.vstack([_embed_batch(preds[i:i+BATCH]) for i in range(0, len(preds), BATCH)])
        er = np.vstack([_embed_batch(refs [i:i+BATCH]) for i in range(0, len(refs),  BATCH)])
        return _mean_cosine(ep, er)
    except ImportError:
        log.warning("openai package not installed — skip.")
        return {}
    except Exception as e:
        log.warning(f"OpenAI embedding failed: {e}")
        return {}

def compute_gemini_embedding(preds: list, refs: list) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        def _embed(texts):
            vecs = []
            for t in tqdm(texts, desc="Gemini embed", leave=False):
                r = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=t,
                    task_type="SEMANTIC_SIMILARITY",
                )
                vecs.append(r["embedding"])
            return np.array(vecs)
        ep = _embed(preds)
        er = _embed(refs)
        return _mean_cosine(ep, er)
    except ImportError:
        log.warning("google-generativeai not installed — skip.")
        return {}
    except Exception as e:
        log.warning(f"Gemini embedding failed: {e}")
        return {}


# ── Corpus metrics ────────────────────────────────────────────────────────────

def build_corpus_metrics(records: list, compute_cosine: bool = True) -> dict:
    n = len(records)

    def mean_f1(metric: str) -> float:
        return round(100 * sum(r["rouge"][metric]["f1"] for r in records) / n, 4)
    def mean_p(metric: str) -> float:
        return round(100 * sum(r["rouge"][metric]["precision"] for r in records) / n, 4)
    def mean_r(metric: str) -> float:
        return round(100 * sum(r["rouge"][metric]["recall"] for r in records) / n, 4)

    corpus = {
        "rouge1_f1":        mean_f1("rouge1"),
        "rouge2_f1":        mean_f1("rouge2"),
        "rouge3_f1":        mean_f1("rouge3"),
        "rougeL_f1":        mean_f1("rougeL"),
        "rouge1_precision": mean_p("rouge1"),
        "rouge1_recall":    mean_r("rouge1"),
        "rouge2_precision": mean_p("rouge2"),
        "rouge2_recall":    mean_r("rouge2"),
        "rouge3_precision": mean_p("rouge3"),
        "rouge3_recall":    mean_r("rouge3"),
        "rougeL_precision": mean_p("rougeL"),
        "rougeL_recall":    mean_r("rougeL"),
        "total_samples":    n,
    }

    preds     = [r["prediction"]  for r in records]
    best_refs = [r["best_target"] for r in records]

    bs = compute_bertscore(preds, best_refs)
    if bs:
        corpus["bertscore_precision"] = bs["mean_precision"]
        corpus["bertscore_recall"]    = bs["mean_recall"]
        corpus["bertscore_f1"]        = bs["mean_f1"]
        for i, rec in enumerate(records):
            rec["bertscore_f1"] = bs["per_sample_f1"][i]

    if compute_cosine:
        log.info("Computing LaBSE cosine similarity …")
        labse = compute_labse(preds, best_refs)
        if labse:
            corpus["cosine_labse"] = labse["mean"]
            for i, rec in enumerate(records):
                rec.setdefault("cosine", {})["labse"] = labse["per_sample"][i]

        log.info("Computing BGE-M3 (LASER) cosine similarity …")
        laser = compute_laser(preds, best_refs)
        if laser:
            corpus["cosine_laser"] = laser["mean"]
            for i, rec in enumerate(records):
                rec.setdefault("cosine", {})["laser"] = laser["per_sample"][i]

        if USE_PAID_EMBEDDINGS:
            log.info("USE_PAID_EMBEDDINGS=True — computing OpenAI + Gemini …")
            openai_res = compute_openai_embedding(preds, best_refs)
            if openai_res:
                corpus["cosine_openai"] = openai_res["mean"]
                for i, rec in enumerate(records):
                    rec.setdefault("cosine", {})["openai"] = openai_res["per_sample"][i]

            gemini_res = compute_gemini_embedding(preds, best_refs)
            if gemini_res:
                corpus["cosine_gemini"] = gemini_res["mean"]
                for i, rec in enumerate(records):
                    rec.setdefault("cosine", {})["gemini"] = gemini_res["per_sample"][i]
        else:
            log.info("USE_PAID_EMBEDDINGS=False — OpenAI & Gemini skipped.")
            corpus["cosine_openai"] = None
            corpus["cosine_gemini"] = None

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
        return {item["idx"]: item for item in items}
    return {}

def save_outputs(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── Evaluation core ───────────────────────────────────────────────────────────

def evaluate_model_shot(
    model: str,
    shot: str,
    dataset: list,
    sample: int | None,
    force: bool,
) -> dict:
    shot_key = SHOT_KEYS[shot]
    out_path = output_path(model, shot)
    res_path = result_path(model, shot)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    existing  = {} if force else load_checkpoint(out_path)
    data      = dataset[:sample] if sample else dataset

    log.info("=" * 65)
    log.info(f"  Model : {model}  |  Shot : {shot}")
    log.info(f"  Items : {len(data)}"
             + (f"  (resuming from {len(existing)})" if existing else ""))
    log.info("=" * 65)

    records  = list(existing.values())
    done_ids = set(existing.keys())
    pending  = [item for item in data if item["idx"] not in done_ids]

    if pending:
        pbar = tqdm(pending, desc=f"{model} [{shot}]", unit="item")
        for item in pbar:
            raw     = ollama_generate(model, item[shot_key])
            pred    = parse_response(raw, model)
            targets = item["targets"]

            rouge_scores = best_rouge(pred, targets)
            best_tgt     = max(
                targets,
                key=lambda t: rouge_l(tokenize(pred), tokenize(t))["f1"]
            )

            records.append({
                "idx":          item["idx"],
                "source":       item["source"],
                "idiom":        item["idiom"],
                "targets":      targets,
                "best_target":  best_tgt,
                "raw_response": raw,
                "prediction":   pred,
                "rouge":        rouge_scores,
                "cosine":       {},
            })
            done_ids.add(item["idx"])
            save_outputs(out_path, records)
        pbar.close()

    log.info(f"Computing corpus metrics for {model} [{shot}] …")
    corpus = build_corpus_metrics(records, compute_cosine=True)
    save_outputs(out_path, records)

    result_payload = {"model": model, "shot": shot, "corpus_metrics": corpus}
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    _log_result(model, shot, corpus)
    return result_payload


# ── Cosine-only mode ──────────────────────────────────────────────────────────

def run_cosine_only(models: list, shots: list, sample: int | None) -> list:
    all_results = []
    for model in models:
        for shot in shots:
            out_path = output_path(model, shot)
            res_path = result_path(model, shot)
            if not out_path.exists():
                log.warning(f"No output file for {model} [{shot}] — skip")
                continue

            log.info(f"[cosine-only]  {model}  [{shot}]")
            with open(out_path, encoding="utf-8") as f:
                records = json.load(f)

            if sample:
                records = records[:sample]

            preds     = [r["prediction"]  for r in records]
            best_refs = [r["best_target"] for r in records]

            labse = compute_labse(preds, best_refs)
            if labse:
                for i, rec in enumerate(records):
                    rec.setdefault("cosine", {})["labse"] = labse["per_sample"][i]

            laser = compute_laser(preds, best_refs)
            if laser:
                for i, rec in enumerate(records):
                    rec.setdefault("cosine", {})["laser"] = laser["per_sample"][i]

            if USE_PAID_EMBEDDINGS:
                openai_res = compute_openai_embedding(preds, best_refs)
                if openai_res:
                    for i, rec in enumerate(records):
                        rec.setdefault("cosine", {})["openai"] = openai_res["per_sample"][i]

                gemini_res = compute_gemini_embedding(preds, best_refs)
                if gemini_res:
                    for i, rec in enumerate(records):
                        rec.setdefault("cosine", {})["gemini"] = gemini_res["per_sample"][i]

            save_outputs(out_path, records)

            def _safe_mean(key):
                vals = [r.get("cosine", {}).get(key) for r in records]
                vals = [v for v in vals if v is not None]
                return round(float(np.mean(vals)), 6) if vals else None

            corpus = {}
            if res_path.exists():
                with open(res_path, encoding="utf-8") as f:
                    prev = json.load(f)
                corpus = prev.get("corpus_metrics", {})

            corpus["cosine_labse"]  = _safe_mean("labse")
            corpus["cosine_laser"]  = _safe_mean("laser")
            corpus["cosine_openai"] = _safe_mean("openai")
            corpus["cosine_gemini"] = _safe_mean("gemini")

            result_payload = {"model": model, "shot": shot, "corpus_metrics": corpus}
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False, indent=2)

            _log_result(model, shot, corpus)
            all_results.append(result_payload)

    return all_results


# ── Metrics-only mode ─────────────────────────────────────────────────────────

def run_metrics_only(models: list, shots: list, sample: int | None) -> list:
    all_results = []
    for model in models:
        for shot in shots:
            out_path = output_path(model, shot)
            res_path = result_path(model, shot)
            if not out_path.exists():
                log.warning(f"No output file for {model} [{shot}] — skip")
                continue

            with open(out_path, encoding="utf-8") as f:
                records = json.load(f)

            if sample:
                records = records[:sample]

            for rec in records:
                if "rouge" not in rec:
                    rec["rouge"] = best_rouge(rec["prediction"], rec["targets"])
                if "best_target" not in rec:
                    rec["best_target"] = max(
                        rec["targets"],
                        key=lambda t: rouge_l(tokenize(rec["prediction"]), tokenize(t))["f1"]
                    )

            corpus = build_corpus_metrics(records, compute_cosine=True)
            save_outputs(out_path, records)

            result_payload = {"model": model, "shot": shot, "corpus_metrics": corpus}
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False, indent=2)

            _log_result(model, shot, corpus)
            all_results.append(result_payload)

    return all_results


# ── Summary and logging ───────────────────────────────────────────────────────

def _log_result(model: str, shot: str, corpus: dict):
    has_bs  = "bertscore_f1" in corpus
    has_cos = any(corpus.get(f"cosine_{k}") is not None
                  for k in ("labse", "laser", "openai", "gemini"))

    line = (
        f"  ✓ {model} [{shot}]  "
        f"R1={corpus.get('rouge1_f1', float('nan')):.2f}  "
        f"R2={corpus.get('rouge2_f1', float('nan')):.2f}  "
        f"R3={corpus.get('rouge3_f1', float('nan')):.2f}  "
        f"RL={corpus.get('rougeL_f1', float('nan')):.2f}"
    )
    if has_bs:
        line += f"  BS-F1={corpus['bertscore_f1']:.2f}"
    if has_cos:
        for k in ("labse", "laser", "openai", "gemini"):
            v = corpus.get(f"cosine_{k}")
            if v is not None:
                line += f"  {k.upper()}={v:.4f}"
    log.info(line)


def print_summary(all_results: list):
    has_bs   = any("bertscore_f1" in r["corpus_metrics"] for r in all_results)
    cos_keys = [k for k in ("labse", "laser", "openai", "gemini")
                if any(r["corpus_metrics"].get(f"cosine_{k}") is not None
                       for r in all_results)]

    log.info("\n" + "=" * 110)
    log.info("SUMMARY — Task 1: Paraphrase Quality")
    log.info("=" * 110)

    hdr = (f"{'#':<3} {'Model':<30} {'Shot':<8} "
           f"{'R-1':>7} {'R-2':>7} {'R-3':>7} {'R-L':>7}")
    if has_bs:
        hdr += f"  {'BS-P':>7} {'BS-R':>7} {'BS-F1':>7}"
    for k in cos_keys:
        hdr += f"  {k.upper():>8}"
    log.info(hdr)
    log.info("-" * 110)

    for idx, r in enumerate(all_results, 1):
        c   = r["corpus_metrics"]
        row = (f"{idx:<3} {r['model']:<30} {r['shot']:<8} "
               f"{c.get('rouge1_f1', float('nan')):>7.2f} "
               f"{c.get('rouge2_f1', float('nan')):>7.2f} "
               f"{c.get('rouge3_f1', float('nan')):>7.2f} "
               f"{c.get('rougeL_f1', float('nan')):>7.2f}")
        if has_bs:
            row += (f"  {c.get('bertscore_precision', float('nan')):>7.2f}"
                    f"  {c.get('bertscore_recall',    float('nan')):>7.2f}"
                    f"  {c.get('bertscore_f1',        float('nan')):>7.2f}")
        for k in cos_keys:
            v = c.get(f"cosine_{k}")
            row += f"  {v:>8.4f}" if v is not None else f"  {'—':>8}"
        log.info(row)

    log.info("=" * 110)
    log.info(
        "ROUGE values ×100 (F1).  BERTScore ×100.  Cosine = raw (0–1).  "
        f"Paid embeddings: {'ON' if USE_PAID_EMBEDDINGS else 'OFF (set USE_PAID_EMBEDDINGS=True to enable)'}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bangla Idiom Paraphrase Quality Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Normal (inference + all metrics):
    python code/evaluate_paraphrase_quality.py

  Recompute all metrics from saved outputs (no re-inference):
    python code/evaluate_paraphrase_quality.py --metrics-only

  Update only cosine embeddings (after enabling paid APIs in config.yml):
    python code/evaluate_paraphrase_quality.py --cosine-only
        """,
    )
    parser.add_argument("--models",       nargs="+", default=MODELS)
    parser.add_argument("--shots",        nargs="+", default=SHOT_TYPES, choices=SHOT_TYPES)
    parser.add_argument("--sample",       type=int,  default=None,
                        help="Evaluate only first N samples")
    parser.add_argument("--data",         default=DATA_PATH)
    parser.add_argument("--prompts",      default=PROMPT_PATH)
    parser.add_argument("--force",        action="store_true",
                        help="Ignore checkpoints and re-run inference")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics from saved outputs (no inference)")
    parser.add_argument("--cosine-only",  action="store_true",
                        help="Update only cosine similarity metrics from saved outputs")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.cosine_only:
        log.info(f"Mode: cosine-only  (paid={'ON' if USE_PAID_EMBEDDINGS else 'OFF'})")
        all_results = run_cosine_only(args.models, args.shots, args.sample)

    elif args.metrics_only:
        log.info("Mode: metrics-only (reload saved outputs, recompute all metrics)")
        all_results = run_metrics_only(args.models, args.shots, args.sample)

    else:
        log.info("Mode: full inference + metrics")
        dataset = build_dataset(args.data, args.prompts)

        all_results = []
        for model in args.models:
            for shot in args.shots:
                try:
                    r = evaluate_model_shot(
                        model, shot, dataset,
                        sample=args.sample,
                        force=args.force,
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
