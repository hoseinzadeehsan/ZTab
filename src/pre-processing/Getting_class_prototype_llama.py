import ast
import json
import os
import re
import sys
from pathlib import Path

import configargparse
from groq import Groq

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from util import get_valid_types


def extract_and_clean_list_content(output_text):
    match = re.search(r"\[.*?\]", output_text, re.DOTALL)
    if match:
        list_content = match.group(0)
    else:
        items = re.split(r",|\n", output_text.strip())
        items = [item.strip() for item in items if item.strip()]
        list_content = items[:50] if len(items) > 50 else items

    if isinstance(list_content, str):
        try:
            output_list = ast.literal_eval(list_content)
        except (SyntaxError, ValueError):
            output_list = list_content
    else:
        output_list = list_content

    return [clean_example(item) for item in output_list]


def clean_example(example):
    if not isinstance(example, str):
        example = str(example)
    example = example.strip("'\"")
    return example.replace('\\"', '"')


def read_existing_samples(file_path: Path):
    if file_path.exists():
        with open(file_path, "r") as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                print(f"Error: {file_path} is not a valid JSON file.")
                return {}
    return {}


def get_llama_samples(semantic_type, class_lists, client, model_name, temperature=0.0, max_tokens=4096, top_p=1.0):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Your task is to generate examples for the following semantic types: {class_lists}\\n"
                    "Generated examples must belong to only one of the semantic types in terms of semantic."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate only 50 unique real-world examples of semantic type {semantic_type} that can be found in web tables "
                    "covering different formats of presenting examples. Your output must be in the format of a python list and do "
                    "not generate anything else."
                ),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
        stop=None,
    )

    output_text = ""
    for chunk in completion:
        output_text += chunk.choices[0].delta.content or ""

    print(f"Generated output for {semantic_type}:")
    print(output_text)
    print("--------------------")

    cleaned_list = extract_and_clean_list_content(output_text)
    if cleaned_list:
        return cleaned_list

    print(f"Error: No valid list content found for {semantic_type}")
    return []


def resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--data-name", type=str, default="dataset-sota", help="Dataset key used for valid semantic types.")
    parser.add_argument("--model-name", type=str, default="llama-3.1-70b-versatile", help="Groq model name.")
    parser.add_argument("--token", type=str, default="", help="Groq API token.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output instead of resume.")
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/description_llama.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    api_key = args.token or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("Groq API key is required. Pass --token or set GROQ_API_KEY.")

    client = Groq(api_key=api_key)

    valid_types = get_valid_types(args.data_name)
    class_lists = ", ".join(valid_types)
    print(f"Valid semantic types ({len(valid_types)}): {valid_types}")

    output_file = resolve_data_path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_semantic_samples = {} if args.overwrite else read_existing_samples(output_file)

    for sem_type in valid_types:
        samples_list = get_llama_samples(
            sem_type,
            class_lists,
            client,
            args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        all_semantic_samples[sem_type] = samples_list

    with open(output_file, "w") as json_file:
        json.dump(all_semantic_samples, json_file, indent=4)

    print(f"All semantic type samples saved to {output_file}")


if __name__ == "__main__":
    main()
