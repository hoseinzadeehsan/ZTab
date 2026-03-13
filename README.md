# ZTab: Domain-based Zero-shot Annotation for Table Columns

This repository is the source code of **"[ZTab: Domain-based Zero-shot Annotation for Table Columns](https://arxiv.org/pdf/2603.11436)"** published at **ICDE 2026**.

---

## ◆ About ZTab

ZTab is a domain-based zero-shot framework for column type annotation on header-less tables, built for cases where labeled tables are unavailable due to privacy constraints or because labeling is time/cost prohibitive. Instead of asking users to share potentially sensitive labeled tables, ZTab takes a schema-level domain specification (a semantic type inventory and example table schemas), generates class prototypes, builds pseudo-tables from schemas and prototypes, and fine-tunes an annotation LLM for robust generalization in in-domain, cross-domain, and cross-ontology settings.

It supports two variants:
- **ZTab-privacy**: open-source LLMs, local fine-tuning/inference
- **ZTab-performance**: closed-source LLMs, highest accuracy

---

## ◆ Structure of the repository

```text
ZTab/
├── data/                         # Dataset splits, type mappings, prototypes, JSONL files
├── src/
│   ├── in_domain.py              # In-domain training/evaluation (open-source LLMs)
│   ├── cross_domain.py           # Cross-domain training/evaluation (open-source LLMs)
│   ├── cross_ontology.py         # Cross-ontology training/evaluation (open-source LLMs)
│   ├── dataset_llm.py            # Dynamic training dataset and prompt construction
│   ├── model_llm.py              # LLM loading and LoRA setup
│   ├── util.py                   # Type mappings and helper functions
│   ├── gpt_fine_tuning.py        # Closed-source fine-tuning pipeline
│   ├── gpt_inference.py          # Closed-source inference and evaluation
│   └── pre-processing/
│       ├── Generate_openai_jsonl.py
│       ├── Getting_class_prototype_GPT.py
│       └── Getting_class_prototype_llama.py
├── README.md
└── requirements.txt
```

---

## ◆ Environment setup

`[Quick Start]`

```bash
git clone <repository_url>
cd ZTab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This creates an isolated Python environment so package versions do not conflict with other projects on your machine.

---

## ◆ Dataset and data setup

### ▶ 1) Download prepared package (recommended)

Download and extract the prepared `data/` package under the repository root:
- [Google Drive package](https://drive.google.com/file/d/1wMtmr3wLt6cD2r7vdJX67vz9OD79r9D6/view?usp=sharing)

The package includes:
- train/validation/test CSV splits
- `types.json`
- class prototype files (for example `description_gpt.json`)
- dataset JSONL files for closed-source fine-tuning

### ▶ 2) Original dataset sources

- T2D, Limaye, Efthymiou (`dataset-wikipedia`): [SemAIDA](https://github.com/alan-turing-institute/SemAIDA)
- SOTAB: [Web Data Commons SOTAB v2](https://webdatacommons.org/structureddata/sotab/v2/)
- WikiTable (`dataset-turl`): [TURL](https://github.com/sunlab-osu/TURL)

---

## ◆ Model training and evaluation

### ▶ Authentication requirements

- **Open-source LLM scripts (ZTab-privacy)** need Hugging Face token via `--token`.
- **Closed-source LLM scripts (ZTab-performance)** need OpenAI token via `--token` (or `OPENAI_API_KEY`).

### ▶ Open-source LLMs (ZTab-privacy)

#### ▸ In-domain

```bash
python src/in_domain.py --data-name dataset-t2d --token <your_hf_token>
python src/in_domain.py --data-name dataset-limaye --token <your_hf_token>
python src/in_domain.py --data-name dataset-wikipedia --token <your_hf_token>
python src/in_domain.py --data-name dataset-sota --header_ratio 0.025 --token <your_hf_token>
python src/in_domain.py --data-name dataset-sota-small --header_ratio 0.025 --token <your_hf_token>
python src/in_domain.py --data-name dataset-sota-dbpedia --header_ratio 0.025 --token <your_hf_token>
python src/in_domain.py --data-name dataset-turl --header_ratio 0.005 --token <your_hf_token>
```

#### ▸ Cross-domain

```bash
python src/cross_domain.py --data-name dataset-t2d --token <your_hf_token>
```

#### ▸ Cross-ontology

```bash
python src/cross_ontology.py --data-name dataset-sota --token <your_hf_token>
python src/cross_ontology.py --data-name dataset-sota-small --token <your_hf_token>
python src/cross_ontology.py --data-name dataset-sota-dbpedia --token <your_hf_token>
```

### ▶ Closed-source LLMs (ZTab-performance)

#### ▸ Fine-tuning

```bash
python src/gpt_fine_tuning.py \
  --data-name dataset-sota \
  --model-name gpt-4o-mini \
  --token <your_openai_token>
```

#### ▸ Inference

```bash
python src/gpt_inference.py \
  --data_name dataset-sota \
  --token <your_openai_token> \
  --id <your_fine_tuned_model_id>
```

---

## ◆ Important hyper-parameters

| Parameter | Why it matters |
|---|---|
| `--data-name` | Chooses dataset split and label space. |
| `--model-name` | Chooses base model family. |
| `--num_epochs` | Training duration; too low underfits, too high may overfit. |
| `--lr` | Most sensitive optimization parameter. |
| `--batch-size`, `--accumulation_steps` | Together define effective batch size and memory usage. |
| `--max-length` | Prompt/target length budget; affects context and memory. |
| `--row_examples` | Number of values used per column in prompts. |
| `--header_ratio` | Amount of header/table context sampled for training. |
| `--only_class_list` | Ignores table schemas and uses only the class list for synthetic training generation. |
| `--description_type`, `--description_length` | Controls synthetic value source and volume. |
| `--rank` | LoRA adapter capacity (quality vs memory tradeoff). |

---

## ◆ Pre-processing

- `Getting_class_prototype_GPT.py`: generates class prototypes using OpenAI.
- `Getting_class_prototype_llama.py`: generates class prototypes using Groq/Llama.
- `Generate_openai_jsonl.py`: generates per-dataset JSONL files (`data/<dataset>.jsonl`) from `data/description_gpt.json` by default.

---

## ◆ Custom dataset (step-by-step)

### ▶ Required CSV schema

Your train/validation/test CSV files should include:
- `table_id`
- `col_idx`
- `class`
- `data`

### ▶ Integration steps

1. ✓ Add split files into `data/`.
2. ✓ Add valid type list(s) in `data/types.json`.
3. ✓ Update `get_valid_types(...)` in `src/util.py`.
4. ✓ Add dataset path mapping in `src/in_domain.py`.
5. ✓ If using closed-source path, add mapping in:
- `src/gpt_fine_tuning.py`
- `src/gpt_inference.py`
6. ✓ If generating JSONL from table splits, update `DATASET_TO_CSV` in `src/pre-processing/Generate_openai_jsonl.py`.

If you only have class names (no train split), you can still generate JSONL with class-only generation logic in the pre-processing script.
