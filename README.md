# [EACL 2027] To What Extent Do Large Language Models Understand Bangla Idioms?


Mousumi Akter¹², Md. Faiyaz Abdullah Sayeedi³, Nurul Labib Sayeedi⁴, Swakkhar Shatabda³

¹Technical University Dortmund · ²Research Center Trustworthy Data Science and Security · ³BRAC University · ⁴United International University

📄 [Paper (arXiv)](https://arxiv.org/pdf/2609.03410)

---

## Repository Structure

```
├── code/                          # Evaluation scripts
│   ├── evaluate_paraphrase_quality.py   # Task 1: Paraphrase Quality
│   ├── evaluate_span_detection.py       # Task 2: Span Detection
│   ├── evaluate_mcq_sub_1.py            # Task 3a: MCQ — select one meaning
│   └── evaluate_mcq_sub_2.py            # Task 3b: MCQ — select multiple meanings
├── data/                          # Dataset (Bagdhara)
│   ├── Bagdhara_v3_merged.json
│   ├── Bagdhara_v3_merged.csv
│   ├── Bagdhara_v3.json
│   └── Bagdhara_v3_test.json
├── tasks/                         # Prompt files for each task
│   ├── task_1_paraphrase_quality_prompt.json
│   ├── task_2_span_detection_prompt.json
│   ├── mcq_task_3_subtask_1_select_one_meaning_prompt.json
│   └── mcq_task_3_subtask_2_select_multiple_meanings_prompt.json
├── models.yml                     # Model list, profiles, and Ollama config
├── config.yml                     # API key template (fill in your keys)
└── requirements.txt
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd Bangla-Idioms
pip install -r requirements.txt
```

**2. Configure models**

Edit `models.yml` to add or remove models from the default list. Each model must be pulled in Ollama first:
```bash
ollama pull phi4:latest
ollama pull gemma3:27b
ollama pull deepseek-r1:14b
```

**3. (Optional) Add API keys for paid embeddings**

Copy `config.yml` and fill in your keys to enable OpenAI / Gemini embeddings in Task 1:
```yaml
api_keys:
  openai: "sk-..."
  gemini: "AIza..."
```
Then set `USE_PAID_EMBEDDINGS = True` at the top of `code/evaluate_paraphrase_quality.py`.

---

## Running Evaluations

> All scripts are run from the repo root. Results are saved to `results/`.

### Task 1 — Paraphrase Quality
```bash
python code/evaluate_paraphrase_quality.py
```
Options:
```bash
--models phi4:latest gemma3:27b   # run specific models
--shots 0-shot 3-shot             # choose few-shot types
--sample 200                      # evaluate on a subset
--metrics-only                    # recompute metrics from saved outputs
--cosine-only                     # update only cosine similarity metrics
```

### Task 2 — Idiom Span Detection
```bash
python code/evaluate_span_detection.py
```
Options:
```bash
--models phi4:latest
--shots 0-shot 3-shot 5-shot
--sample 200
--metrics-only
```

### Task 3a — MCQ: Select One Meaning
```bash
python code/evaluate_mcq_sub_1.py
```
Options:
```bash
--only deepseek-r1:14b            # run a single model
--reset deepseek-r1:14b           # delete checkpoint and re-run
--sample 1000 --seed 42           # reproducible subset
```

### Task 3b — MCQ: Select Multiple Meanings
```bash
python code/evaluate_mcq_sub_2.py
```
Same options as Task 3a.

---

## Output

Results are written to `results/`:
```
results/
├── paraphrase_quality/
│   ├── <model>_<shot>_outputs.json
│   ├── <model>_<shot>_results.json
│   └── summary_all.json
├── span_detection/
│   └── ...
├── mcq_subtask_1/
│   └── ...
└── mcq_subtask_2/
    └── ...
```

---

## Citation

```bibtex
@misc{akter2026extentlargelanguagemodels,
      title={To What Extent Do Large Language Models Understand Bangla Idioms?}, 
      author={Mousumi Akter and Md. Faiyaz Abdullah Sayeedi and Nurul Labib Sayeedi and Swakkhar Shatabda},
      year={2026},
      eprint={2609.03410},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03410}, 
}
```
