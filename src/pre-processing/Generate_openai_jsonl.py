import json
import random
import sys
from pathlib import Path

import configargparse
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from util import get_valid_types

DATASET_TO_CSV = {
    "turl": "train_turl_data.csv",
    "t2d": "train_t2d_doduo.csv",
    "sota-schema": "sota_train_schema.csv",
    "sota-schema-small": "sota_train_small_schema.csv",
    "sota-dbpedia": "sota_train_dbpedia.csv",
    "limaye": None,
    "wikipedia": None,
}

CLASS_ONLY_DATASETS = {"limaye", "wikipedia"}

UNSAFE_CLASS_MAP = {
    "medicine.drug": "medicine.name",
    "medicine.drug_ingredient": "medicine.ingredient",
    "medicine.disease": "medicine.condition",
    "people.ethnicity": "people.population_group",
    "religion.religious_leader": "religion.clergy",
    "religion.religion": "belief.system",
    "military.military_conflict": "history.conflict",
    "military.military_person": "occupation.military_member",
    "government.election": "civic.event",
    "government.general_election": "civic.event.general",
}

SENSITIVE_CLASSES = set(UNSAFE_CLASS_MAP.keys()) | {"government.election_campaign"}


def sanitize_prompt_and_label(prompt: str, label: str):
    for sensitive, safe in UNSAFE_CLASS_MAP.items():
        prompt = prompt.replace(sensitive, safe)
        label = label.replace(sensitive, safe)
    return prompt, label


def table_population(schema, descriptions, row_examples):
    table_values = []
    for header in schema:
        if header in descriptions and descriptions[header]:
            values = random.sample(descriptions[header], min(row_examples, len(descriptions[header])))
        else:
            values = [""] * row_examples
        table_values.append((header, values))
    return table_values


def prompt_construction(table_values, classes):
    table_str = ""
    for idx, (_, values) in enumerate(table_values, 1):
        table_str += f"Column {idx}: {', '.join(str(v) for v in values)}\\n"

    prompts_labels = []
    for _, (header, target_values) in enumerate(table_values):
        prompt = (
            "These are values of columns in a table. Each column starts with Column: followed by the values of that column. "
            "First, look at all the columns to understand the context of the table.\\n"
            f"{table_str}"
            "Your task is to annotate the Target Column using one semantic type that matches the values of the Target Column "
            "and the context of the table from the following list: " + ", ".join(classes) + ".\\n"
            f"Target Column: {', '.join(str(v) for v in target_values)}\\n"
            "Semantic Type:"
        )
        prompts_labels.append((prompt, header))
    return prompts_labels


def resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_descriptions(description_file: Path):
    with open(description_file, "r") as f:
        return json.load(f)


def build_schemas_from_csv(dataset: str, csv_path: Path, schemas_per_dataset: int):
    df = pd.read_csv(csv_path)

    if dataset == "t2d":
        split_cols = df["table_id"].str.split(" ", expand=True)
        if split_cols.shape[1] >= 2:
            df[["table_id", "col"]] = split_cols.iloc[:, :2]

    if "data" in df.columns:
        df = df.drop(columns=["data"])

    schemas = []
    for _, group in df.groupby("table_id"):
        schema = list(group.sort_values("col_idx")["class"])
        schemas.append(schema)

    if len(schemas) == 0:
        return []

    if len(schemas) >= schemas_per_dataset:
        schemas = random.sample(schemas, schemas_per_dataset)
    else:
        schemas *= (schemas_per_dataset // len(schemas)) + 1
        schemas = schemas[:schemas_per_dataset]
    return schemas


def build_class_only_schemas(valid_types, schemas_per_dataset: int):
    schemas = [[class_name] for class_name in valid_types]
    if len(schemas) == 0:
        return []

    if len(schemas) >= schemas_per_dataset:
        return random.sample(schemas, schemas_per_dataset)

    schemas *= (schemas_per_dataset // len(schemas)) + 1
    return schemas[:schemas_per_dataset]


def build_jsonl_for_dataset(dataset, args, descriptions, output_dir: Path):
    valid_types = get_valid_types(dataset)
    classes = sorted(valid_types)

    dataset_only_class = args.only_class_flag or dataset in CLASS_ONLY_DATASETS

    if dataset_only_class:
        schemas = build_class_only_schemas(valid_types, args.schemas_per_dataset)
    else:
        csv_name = DATASET_TO_CSV[dataset]
        if not csv_name:
            raise ValueError(f"No training csv configured for dataset '{dataset}'.")
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        schemas = build_schemas_from_csv(dataset, csv_path, args.schemas_per_dataset)

    missing = set(classes) - set(c for schema in schemas for c in schema)
    for missing_type in missing:
        schemas.append([missing_type])

    fine_tuning_examples = []
    sensitive_class_counter = {cls: 0 for cls in SENSITIVE_CLASSES}

    for schema in schemas:
        if any(
            label in SENSITIVE_CLASSES and sensitive_class_counter[label] >= args.max_sensitive_examples
            for label in schema
        ):
            continue

        table_values = table_population(schema, descriptions, args.k)
        prompts_labels = prompt_construction(table_values, classes)

        for prompt, label in prompts_labels:
            prompt, label = sanitize_prompt_and_label(prompt, label)

            if label in SENSITIVE_CLASSES:
                if sensitive_class_counter[label] >= args.max_sensitive_examples:
                    continue
                sensitive_class_counter[label] += 1

            example = {
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ]
            }
            fine_tuning_examples.append(example)

    output_path = output_dir / f"{dataset}.jsonl"
    with open(output_path, "w") as f:
        for ex in fine_tuning_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(fine_tuning_examples)} examples to {output_path}")


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--k", type=int, default=3, help="Number of sampled values per column.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["turl", "t2d", "sota-schema", "sota-schema-small", "sota-dbpedia", "limaye", "wikipedia"],
        help="Dataset keys to generate JSONL for.",
    )
    parser.add_argument("--schemas-per-dataset", type=int, default=1000, help="Schemas sampled per dataset.")
    parser.add_argument("--max-sensitive-examples", type=int, default=1, help="Max examples for sensitive classes.")
    parser.add_argument("--only-class-flag", action="store_true", default=False, help="Use class-only schemas for all datasets.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are an expert at annotating column types in tables.",
        help="System prompt used in JSONL examples.",
    )
    parser.add_argument(
        "--description-file",
        type=str,
        default="data/description_gpt.json",
        help="Description json path. Default keeps previous behavior.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory where <dataset>.jsonl files are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    for dataset in args.datasets:
        if dataset not in DATASET_TO_CSV:
            raise ValueError(f"Unsupported dataset key: {dataset}. Supported: {list(DATASET_TO_CSV.keys())}")

    description_file = resolve_data_path(args.description_file)
    if not description_file.exists():
        raise FileNotFoundError(f"Description file not found: {description_file}")
    descriptions = load_descriptions(description_file)

    output_dir = resolve_data_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        build_jsonl_for_dataset(dataset, args, descriptions, output_dir)


if __name__ == "__main__":
    main()
