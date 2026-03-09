import ast
import json
import os
import re
import sys
from pathlib import Path

import configargparse
import numpy as np
import openai

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from util import get_valid_types


def extract_and_clean_list_content(output_text):
    output_text = output_text.strip().strip("```python").strip("```")

    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if match:
        list_content = match.group(0)
    else:
        items = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', output_text.strip())
        items = [item.strip().strip("[]") for item in items if item.strip()]
        list_content = items[:50] if len(items) > 50 else items

    if isinstance(list_content, str):
        try:
            output_list = ast.literal_eval(list_content)
        except (SyntaxError, ValueError):
            output_list = [item.strip("'\"[] \n") for item in list_content.split(",") if item.strip("'\"[] \n")]
    else:
        output_list = list_content

    clean_output_list = [clean_example(item) for item in output_list if item and len(str(item).strip()) > 1]
    return list(dict.fromkeys(clean_output_list))[:50]


def clean_example(example):
    if not isinstance(example, str):
        example = str(example)
    return example.strip("'\"").replace('\\"', '"').strip()


def read_existing_samples(file_path: Path):
    if file_path.exists():
        with open(file_path, "r") as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                print(f"Error: {file_path} is not a valid JSON file.")
                return {}
    return {}


def validate_examples(examples, semantic_type, valid_types, client):
    if not examples:
        return []

    try:
        embedding_model = "text-embedding-3-small"
        target_emb = client.embeddings.create(input=semantic_type, model=embedding_model).data[0].embedding
        class_embs = {
            c: client.embeddings.create(input=c, model=embedding_model).data[0].embedding
            for c in valid_types
            if c != semantic_type
        }

        valid_examples = []
        for ex in examples[:50]:
            ex_emb = client.embeddings.create(input=ex, model=embedding_model).data[0].embedding
            target_sim = np.dot(target_emb, ex_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(ex_emb))
            other_sims = [
                np.dot(class_embs[c], ex_emb) / (np.linalg.norm(class_embs[c]) * np.linalg.norm(ex_emb))
                for c in class_embs
            ]
            if target_sim > max(other_sims, default=0) + 0.1:
                valid_examples.append(ex)

        return valid_examples[:50]
    except Exception as e:
        print(f"Embedding validation failed: {e}. Returning unfiltered examples.")
        return examples[:50]


def get_gpt_samples(
    semantic_type,
    valid_types,
    class_lists,
    client,
    model_name,
    max_retries=3,
    temperature=0.2,
    max_tokens=4096,
    top_p=0.9,
    use_embedding_validation=False,
):
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at generating realistic examples for semantic types in web tables. "
                            f"The valid semantic types are: {class_lists}. "
                            "Generated examples must strictly belong to the requested semantic type and must not overlap "
                            "with or be confusable with other semantic types in the list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Generate exactly 50 unique real-world examples of semantic type '{semantic_type}' that can be found in web tables, "
                            "covering different formats of presenting examples. Your output must be a Python list "
                            "(e.g., ['item1', 'item2', ...]) and nothing else."
                        ),
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            output_text = completion.choices[0].message.content
            print(f"Attempt {attempt} - Generated output for {semantic_type}:")
            print(output_text)
            print("--------------------")

            cleaned_list = extract_and_clean_list_content(output_text)
            if use_embedding_validation:
                validated_list = validate_examples(cleaned_list, semantic_type, valid_types, client)
            else:
                validated_list = cleaned_list

            if validated_list:
                print(f"Success: Valid examples generated for {semantic_type} on attempt {attempt}")
                return validated_list

            print(f"Warning: No valid list content found for {semantic_type} on attempt {attempt}. Retrying...")
        except Exception as e:
            print(f"Error on attempt {attempt} for {semantic_type}: {e}. Retrying...")

    print(f"Error: Failed to generate valid examples for {semantic_type} after {max_retries} attempts.")
    return []


def resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--data-name", type=str, default="turl", help="Dataset key used for valid semantic types.")
    parser.add_argument("--model-name", type=str, default="gpt-4o", help="OpenAI model for prototype generation.")
    parser.add_argument("--token", type=str, default="", help="OpenAI API token.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per semantic type.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens.")
    parser.add_argument("--use-embedding-validation", action="store_true", default=False, help="Enable embedding filter.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output instead of resume.")
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Output JSON file path. Default: data/description_<model-name>.json",
    )
    args = parser.parse_args()

    api_key = args.token or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is required. Pass --token or set OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = api_key
    client = openai.OpenAI()

    valid_types = get_valid_types(args.data_name)
    class_lists = ", ".join(valid_types)
    print(f"Valid semantic types ({len(valid_types)}): {valid_types}")

    output_file = (
        resolve_data_path(args.output_file) if args.output_file else DATA_DIR / f"description_{args.model_name}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_semantic_samples = {} if args.overwrite else read_existing_samples(output_file)

    for sem_type in valid_types:
        samples_list = get_gpt_samples(
            sem_type,
            valid_types,
            class_lists,
            client,
            args.model_name,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            use_embedding_validation=args.use_embedding_validation,
        )
        all_semantic_samples[sem_type] = samples_list

    with open(output_file, "w") as json_file:
        json.dump(all_semantic_samples, json_file, indent=4)

    print(f"All semantic type samples saved to {output_file}")


if __name__ == "__main__":
    main()
