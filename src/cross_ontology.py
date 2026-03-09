from transformers import  default_data_collator, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import pandas as pd
from huggingface_hub import login
from sklearn.metrics import classification_report, f1_score
import configargparse
from torch.utils.data import DataLoader, RandomSampler
from util import get_valid_types
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from torch.cuda.amp import GradScaler
from torch import autocast
from dataset_llm import TrainScenario2DynamicDataset
import copy
import gc
from model_llm import load_model_tokenizer_and_peft_config
import os

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings("ignore",
                        message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.")


def keep_first_five(values, row_examples):

    if args.data_name == 'dataset-t2d':
        return values[:2000]
    if args.data_name == 'dataset-sota-dbpedia':
        return ", ".join(values.split(" [SEP] ")[:row_examples])[:500]
    return ", ".join(values.split(" [SEP] ")[:row_examples])[:1500]

# Data loading
def load_data(data_csv_file, row_examples):
    data_df = pd.read_csv(data_csv_file)
    data_df['data'] = data_df['data'].apply(keep_first_five, row_examples=row_examples)

    return data_df

def create_table_wise_prompt_column(data_df, text_column, label_column, id_column, pre_prompt, after_prompt, tokenizer,
                                    max_length):
    grouped = data_df.groupby('table_id')
    rows = []

    # Construct table-wise prompts
    for table_id, group in grouped:
        base_prompt = ""
        # group = group.sort_values(by='col_idx')
        for idx, row in group.iterrows():
            base_prompt += f"Column {row[id_column]}: {row[text_column]}. \n"
        base_prompt = base_prompt.rstrip(" \n")
        base_prompt = f"{pre_prompt} \n{base_prompt}. \n{after_prompt} "

        for idx, row in group.iterrows():
            final_promt = f"{base_prompt} \nTarget Column: {row[text_column]} \nSemantic Type: "
            tokenized_prompt = tokenizer(final_promt, truncation=False, return_tensors='pt')
            tokenized_length = tokenized_prompt.input_ids.shape[1]

            # Ensure the prompt length is within max_length
            if tokenized_length > max_length:
                # print(final_promt)
                print('Error in test of val', tokenized_length)
            rows.append({
                id_column: row[id_column],
                "table_id": table_id,
                text_column: final_promt,
                label_column: row[label_column]
            })

    # Create a new DataFrame with the constructed prompts and class values
    result_df = pd.DataFrame(rows)

    return result_df

# Training function
def train_model(model, optimizer, lr_scheduler, train_dataloader, device, num_epochs, accumulation_steps, validation_df,
                test_df):
    scaler = GradScaler()
    best_model = copy.deepcopy(model)
    best_val_acc = 0
    best_test_acc = 0
    train_loss_all = []
    test_acc_all = []
    validation_acc_all = []
    for epoch in range(num_epochs):
        train_dataset.select_tables()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type='cuda'):
                # with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, step {step}. Stopping training.")
                return model
            total_loss += loss.detach().float()
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

        if len(train_dataloader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss = total_loss / len(train_dataloader)
        train_loss_value = train_loss.item()

        torch.cuda.empty_cache()
        del batch, outputs, loss
        gc.collect()

        validation_report_df = evaluate_model(model, validation_df)
        validation_accuracy = validation_report_df.loc['accuracy'].iloc[0]
        test_report_df = evaluate_model(model, test_df)
        test_accuracy = test_report_df.loc['accuracy'].iloc[0]

        if validation_accuracy > best_val_acc:
            best_val_acc = validation_accuracy
            best_test_acc = test_accuracy
            best_model = copy.deepcopy(model).cpu()
            print('--------------Best model has changed---------')

        train_loss_all.append(train_loss_value)
        validation_acc_all.append(validation_accuracy)
        test_acc_all.append(test_accuracy)
        print(f"{epoch=}: {train_loss_value=} {validation_accuracy=} {test_accuracy=}")

    for i in range(num_epochs):
        print(
            f"epoch: {i}, train_loss_value: {train_loss_all[i]}, validation_accuracy: {validation_acc_all[i]}, test_accuracy: {test_acc_all[i]}")

    print('-------results of best model recorded during training---------')
    print(f"{best_val_acc=} {best_test_acc=}")
    print('------------------------')

    return best_model


def evaluate_model(model, data_df):
    torch.cuda.empty_cache()
    results = []
    true_labels = []
    generated_texts = []

    batch_size = 8  # Adjust this based on your GPU memory

    # for start_idx in tqdm(range(0, len(data_df), batch_size)):
    for start_idx in range(0, len(data_df), batch_size):
        end_idx = min(start_idx + batch_size, len(data_df))
        batch_df = data_df.iloc[start_idx:end_idx]

        values = batch_df["data"].tolist()
        true_labels.extend(batch_df["class"].tolist())

        prompts = [f"{x}" for x in values]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2400).to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                     max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

            batch_generated_texts = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

            for i, value in enumerate(values):
                prompt_len = len(f"{value}")
                generated_texts.append(batch_generated_texts[i][prompt_len:].strip())
                results.append((batch_generated_texts[i][prompt_len:].strip(), batch_df["class"].iloc[i]))

    # generated_texts_before_mapping = generated_texts
    generated_texts = map_to_closest_class(generated_texts)

    generated_classification_report = classification_report(true_labels, generated_texts, output_dict=True)
    generated_micro_f1 = f1_score(true_labels, generated_texts, average='micro')
    generated_classification_report['micro avg'] = {'precision': generated_micro_f1, 'recall': generated_micro_f1,
                                                    'f1-score': generated_micro_f1, 'support': len(true_labels)}

    # Convert classification reports to dataframes and save as CSV
    generated_report_df = pd.DataFrame(generated_classification_report).transpose()

    return generated_report_df

def map_to_closest_class(generated_labels):
    generated_input_ids = tokenizer(generated_labels, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"].to(device)
    generated_embeddings = initial_model.get_input_embeddings()(generated_input_ids).mean(dim=1)

    mapped_labels = []
    for gen_embed in generated_embeddings:
        similarities = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), class_embeddings, dim=1)
        mapped_labels.append(class_list_test[similarities.argmax().item()])
    return mapped_labels


parser = configargparse.ArgParser()
# Configuration

parser.add_argument("--gpu", type=str, default="cuda",
                    help="Specify which GPU to use (e.g., 'cuda' or 'cpu').")
parser.add_argument('--max-length', type=int, default=1600,
                    help="Maximum sequence length for processing inputs.")
parser.add_argument('--row_examples', type=int, default=3,
                    help="Number of row samples per column for pseudo-table generation.")
parser.add_argument('--batch-size', type=int, default=1,
                    help="Batch size for training, validation, and testing.")
parser.add_argument('--accumulation_steps', type=int, default=8,
                    help="Number of gradient accumulation steps to effectively increase batch size.")
parser.add_argument('--data-name', type=str, default='dataset-sota',
                    help="Dataset name for loading training and evaluation data.")
parser.add_argument('--mode', type=str, default='train',
                    help="Execution mode: 'train' for training, 'eval' for evaluation.")
parser.add_argument('--model-name', type=str, default='Qwen7',
                    help="Name of the large language model (LLM) to use.")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning rate for training.")
parser.add_argument("--dropout_prob", type=float, default=0.01,
                    help="Dropout probability to prevent overfitting.")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="Total number of training epochs.")
parser.add_argument("--quantized", action="store_true", default=True,
                    help="Enable model quantization for reduced memory usage and efficiency.")
parser.add_argument("--only_class_list", action="store_true", default=False,
                    help="Use only the class list for training without additional data.")
parser.add_argument("--original_label", action="store_true", default=False,
                    help="Preserve original labels without applying label remapping.")
parser.add_argument("--description_type", nargs="+", default=["llama"], type=str,
                    help="Source of class descriptions.")
parser.add_argument("--header_ratio", type=float, default=0.025,
                    help="Ratio of headers used for training in table-based datasets.")
parser.add_argument('--remapping-step', type=str, default='prompt',
                    help="Method for label remapping: 'embedding' or 'prompt'.")
parser.add_argument("--rank", default=256, type=int,
                    help="Rank of the LoRA (Low-Rank Adaptation) model used for fine-tuning.")
parser.add_argument("--description_length", type=float, default=-1,
                    help="Length of class descriptions used for pseudo-table generation (-1 for default setting).")
parser.add_argument('--token', type=str, default='',
                    help="Hugginface access token.")
args = parser.parse_args()

device = torch.device(args.gpu)
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
print(args.model_name)

access_token = args.token
login(access_token)

if args.data_name == 'dataset-sota-small':
    train_csv_file = os.path.join(data_dir, "sota_train_small_schema.csv")
elif args.data_name == 'dataset-sota-dbpedia':
    train_csv_file = os.path.join(data_dir, "sota_train_dbpedia.csv")
elif args.data_name == 'dataset-sota':
    train_csv_file = os.path.join(data_dir, "sota_train_schema.csv")
val_csv_file = os.path.join(data_dir, "sota_validation_dbpedia.csv")
test_csv_file = os.path.join(data_dir, "sota_test_dbpedia.csv")


text_column = "data"
label_column = "class"
max_length = args.max_length
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
row_examples = args.row_examples
quantized = args.quantized
class_list = get_valid_types(args.data_name)
class_list_str = ', '.join(class_list)
class_list_test = get_valid_types('dataset-sota-dbpedia')
class_list_test_str = ', '.join(class_list_test)
only_class_list = args.only_class_list
remapping_step = args.remapping_step

validation_df = load_data(val_csv_file, row_examples)
test_df = load_data(test_csv_file, row_examples)
validation_df['data'] = validation_df['data'].replace('NaN', '', regex=True)
test_df['data'] = test_df['data'].replace('NaN', '', regex=True)

# init_instruction_pre_table = f'These are values columns in a table. Each column start with Column: followed by values of that column.\nFirst look at all the columns to get the context of the table.'
init_instruction_post_column = f'\nYour task is to annotate Target Column using only one semantic type that matches the values of Target Column and context of the table from the following list: {class_list_str}.'
if remapping_step == 'embedding':
    init_instruction_post_column_test = f'\nYour task is to annotate Target Column using only one semantic type that matches the values of Target Column and context of the table from the following list: {class_list_str}.'
elif remapping_step == 'prompt':
    init_instruction_post_column_test = f'\nYour task is to annotate Target Column using only one semantic type that matches the values of Target Column and context of the table from the following list: {class_list_test_str}.'

model, tokenizer, initial_model = load_model_tokenizer_and_peft_config(args, access_token, device, quantized)
print(model.print_trainable_parameters())
init_instruction_pre_table = f'These are values of columns in a table. Each column start with Column: followed by values of that column.\nFirst look at all the columns to get the context of the table.'
test_df = create_table_wise_prompt_column(test_df, text_column, label_column, 'col_idx', init_instruction_pre_table,
                                          init_instruction_post_column_test, tokenizer, max_length)
validation_df = create_table_wise_prompt_column(validation_df, text_column, label_column, 'col_idx',
                                                init_instruction_pre_table, init_instruction_post_column_test,
                                                tokenizer, max_length)

class_input_ids = tokenizer(class_list_test, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
class_embeddings = initial_model.get_input_embeddings()(class_input_ids).mean(dim=1)

train_dataset = TrainScenario2DynamicDataset(tokenizer, init_instruction_pre_table, init_instruction_post_column,
                                             args.data_name, None, max_length, row_examples, device, args.description_type,
                                             args.description_length, args.header_ratio, only_class_list)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              # sampler=train_sampler,
                              batch_size=batch_size,
                              collate_fn=default_data_collator,
                              shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)

model = train_model(model, optimizer, lr_scheduler, train_dataloader, device, num_epochs, accumulation_steps,
                    validation_df, test_df)
