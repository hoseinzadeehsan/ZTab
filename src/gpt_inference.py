import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import configargparse
import openai
import os
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import warnings
from util import get_valid_types
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Configuration
parser = configargparse.ArgParser()
parser.add_argument('--data_name', type=str, default='dataset-limaye', help="Name of the dataset to load for evaluation.")
parser.add_argument('--max_length', type=int, default=4096, help="Maximum sequence length for processing input data.")
parser.add_argument('--row_examples', type=int, default=3,
                    help="Number of row examples used when constructing pseudo-tables.")
parser.add_argument('--original_label', action="store_true", default=False,
                    help="Use original labels without label remapping.")
parser.add_argument('--id', type=str, default=None,
                    help="ID of the fine-tuned GPT model (e.g., from fine_tune_result_<job_id>.json).")
parser.add_argument('--token', type=str, default='',
                    help="Hugginface access token.")

args = parser.parse_args()
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = args.token
client = openai.OpenAI()


# Function to preprocess data
def keep_first_five(values, row_examples):
    if args.data_name in ['dataset-t2d', 'dataset-limaye', 'dataset-wikipedia']:
        return values[:2000]
    return ", ".join(values.split(" [SEP] ")[:row_examples])[:1500]


def load_data(data_csv_file, row_examples):
    data_df = pd.read_csv(data_csv_file)
    data_df['data'] = data_df['data'].apply(keep_first_five, row_examples=row_examples)
    data_df['data'] = data_df['data'].replace('NaN', '', regex=True)
    # print(data_df)
    return data_df


# Function to create table-wise prompts
def create_table_wise_prompt_column(data_df, text_column, label_column, id_column, pre_prompt, after_prompt,
                                    max_length):
    grouped = data_df.groupby('table_id')
    rows = []
    for table_id, group in grouped:
        base_prompt = ""
        for idx, row in group.iterrows():
            base_prompt += f"Column {row[id_column]}: {row[text_column]}. \n"
        base_prompt = base_prompt.rstrip(" \n")
        base_prompt = f"{pre_prompt} \n{base_prompt}. \n{after_prompt} "
        for idx, row in group.iterrows():
            final_prompt = f"{base_prompt} \nTarget Column: {row[text_column]} \nSemantic Type: "
            if len(final_prompt) > max_length:
                print(
                    f"Warning: Prompt length {len(final_prompt)} exceeds max_length {max_length} for table {table_id}")
            rows.append({
                id_column: row[id_column],
                "table_id": table_id,
                text_column: final_prompt,
                label_column: row[label_column]
            })
    return pd.DataFrame(rows)


# Function to map generated labels to closest class using embeddings
def map_to_closest_class(generated_labels, class_list, class_embeddings, model, device):
    encoded_labels = model.encode(generated_labels, convert_to_tensor=True, device=device)
    mapped_labels = []
    for gen_embed in encoded_labels:
        similarities = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), class_embeddings, dim=1)
        mapped_labels.append(class_list[similarities.argmax().item()])
    return mapped_labels


# Evaluation function using fine-tuned GPT model
def evaluate_model(data_df, model_id, class_list, original_label, device):
    # Load sentence transformer for embedding-based label mapping
    embed_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)



    # Compute embeddings for class list
    class_embeddings = embed_model.encode(class_list, convert_to_tensor=True, device=device)

    results = []
    true_labels = []
    generated_texts = []
    batch_size = 8  # Adjust based on API rate limits

    for start_idx in range(0, len(data_df), batch_size):
        end_idx = min(start_idx + batch_size, len(data_df))
        batch_df = data_df.iloc[start_idx:end_idx]
        prompts = batch_df["data"].tolist()
        true_labels.extend(batch_df["class"].tolist())

        # Process prompts one at a time to ensure single ChatCompletion response
        batch_generated_texts = []
        for prompt in prompts:
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,  # Short output for semantic type
                    temperature=0.2, #0.2,
                    top_p= 0.9 #0.9
                )
                generated_text = completion.choices[0].message.content.strip()
                batch_generated_texts.append(generated_text)
            except Exception as e:
                print(f"Error in API call for prompt: {e}")
                print(f"Prompt: {prompt[:100]}...")  # Truncate for readability
                print(f"Response: {completion}")  # Log full response for debugging
                batch_generated_texts.append('')  # Fallback empty response

        generated_texts.extend(batch_generated_texts)
        for i, gen_text in enumerate(batch_generated_texts):
            results.append((gen_text, batch_df["class"].iloc[i]))

    # Map to closest class if original_label=False
    if not original_label:
        generated_texts = map_to_closest_class(generated_texts, class_list, class_embeddings, embed_model, device)

    # Compute classification report and micro F1
    generated_classification_report = classification_report(true_labels, generated_texts, output_dict=True)
    generated_micro_f1 = f1_score(true_labels, generated_texts, average='micro')
    generated_classification_report['micro avg'] = {
        'precision': generated_micro_f1,
        'recall': generated_micro_f1,
        'f1-score': generated_micro_f1,
        'support': len(true_labels)
    }

    # Convert to DataFrame and save
    generated_report_df = pd.DataFrame(generated_classification_report).transpose()
    output_file = f"evaluation_report_{args.data_name}_{model_id.replace(':', '_')}.csv"
    generated_report_df.to_csv(output_file)
    labels = np.unique(true_labels)
    cm = confusion_matrix(true_labels, generated_texts, labels=labels)
    # Normalize the confusion matrix over the true (row) axis to make each row sum to 1
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Create a DataFrame from the normalized confusion matrix
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    # Calculate the support for each class
    support = pd.Series(true_labels).value_counts().reindex(labels, fill_value=0).values
    # Add the support column to the DataFrame
    cm_df['support'] = support
    output_file_conf = f"confusion_matrix_{args.data_name}_{model_id.replace(':', '_')}.csv"
    cm_df.to_csv(output_file_conf)
    print(f"Evaluation results saved to {output_file}")

    return generated_report_df


# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_list = get_valid_types(args.data_name)
class_list_str = ', '.join(class_list)


# Load dataset paths
if args.data_name == 'dataset-t2d':
    test_csv_file = os.path.join(data_dir, "test_t2d_doduo.csv")
elif args.data_name in ['dataset-sota', 'dataset-sota-corner']:
    test_csv_file = os.path.join(data_dir, "sota_test_schema.csv")
elif args.data_name == 'dataset-sota-dbpedia':
    test_csv_file = os.path.join(data_dir, "sota_test_dbpedia.csv")
elif args.data_name == 'dataset-limaye':
    test_csv_file = os.path.join(data_dir, "test_limaye_doduo.csv")
elif args.data_name == 'dataset-wikipedia':
    test_csv_file = os.path.join(data_dir, "test_wikipedia_doduo.csv")
elif args.data_name == 'dataset-turl':
    test_csv_file = os.path.join(data_dir, "test_turl_data.csv")
else:
    raise ValueError(f"Unsupported data_name: {args.data_name}")

# Load test data
test_df = load_data(test_csv_file, args.row_examples)
if args.data_name == "dataset-t2d":
    test_df[['table_id', 'col']] = test_df['table_id'].str.split(' ', expand=True)

test_df = test_df[test_df['class'].isin(class_list)]

# Define prompts
init_instruction_pre_table = (
    "These are values of columns in a table. Each column starts with Column: followed by values of that column.\n"
    "First look at all the columns to get the context of the table."
)
init_instruction_post_column = (
    f"\nYour task is to annotate Target Column using only one semantic type that matches the values of Target Column "
    f"and context of the table from the following list: {class_list_str}."
)

# Create table-wise prompts for test data
test_df = create_table_wise_prompt_column(
    test_df, 'data', 'class', 'col_idx', init_instruction_pre_table,
    init_instruction_post_column, args.max_length
)

model_id = args.id
report_df = evaluate_model(test_df, model_id, class_list, args.original_label, device)
print(report_df)
