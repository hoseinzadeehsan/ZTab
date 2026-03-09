import torch
from torch.utils.data import Dataset
from util import get_description, get_valid_types
import random
import pandas as pd
import uuid
import os

class TrainScenario2DynamicDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 init_instruction: str,
                 post_instruction: str,
                 data_name: str,
                 train_csv_path: str = None,
                 max_length: int = 1600,
                 row_examples: int = 3,
                 device: torch.device = None,
                 description_type: str = ['llama'],
                 description_length: int = -1,
                 header_ratio: float = 1.0,
                 only_class_list: bool = False):

        self.row_examples = row_examples
        self.post_instruction = post_instruction
        self.init_instruction = init_instruction
        self.header_ratio = header_ratio
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.only_class_list = only_class_list

        valid_types_task = get_valid_types(data_name)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

        if train_csv_path is not None:
            self.data = pd.read_csv(train_csv_path)
        else:
            # Backward-compatible fallback mapping.
            if data_name == 'dataset-t2d' or data_name == 'dataset-wikipedia' or data_name == 'dataset-limaye':
                self.data = pd.read_csv(os.path.join(data_dir, 'train_t2d_doduo.csv'))
            elif data_name == 'dataset-sota':
                self.data = pd.read_csv(os.path.join(data_dir, 'sota_train_schema.csv'))
            elif data_name == 'dataset-sota-small':
                self.data = pd.read_csv(os.path.join(data_dir, 'sota_train_small_schema.csv'))
            elif data_name == 'dataset-sota-dbpedia':
                self.data = pd.read_csv(os.path.join(data_dir, 'sota_train_dbpedia.csv'))
            else:
                raise ValueError("data_name is not valid")

        self.data = self.data[self.data['class'].isin(valid_types_task)]
        if len(self.data) > 0 and 'table_id' in self.data.columns and isinstance(self.data['table_id'].iloc[0], str) and ' ' in self.data['table_id'].iloc[0]:
            split_cols = self.data['table_id'].str.split(' ', expand=True)
            if split_cols.shape[1] >= 2:
                self.data[['table_id', 'col']] = split_cols.iloc[:, :2]

        if only_class_list:
            self.data = self.data.drop(self.data.index)

        # Prepare descriptions
        self.descriptions = get_description(data_name, description_type, description_length)
        total_samples = []
        for key, value in self.descriptions.items():
            total_samples.append(len(value))
        print('Average samples', sum(total_samples) / len(total_samples))
        present_types = self.data['class'].unique()
        missing_types = [t for t in valid_types_task if t not in present_types]

        print('Missing types', missing_types)

        # Add missing types as new rows
        self.add_missing_types(missing_types, 1)

        self.data['data'] = self.data['class'].map(self.descriptions)
        self.table_ids = list(self.data['table_id'].unique())
        self.original_data = self.data.copy()  # Keep a copy of the original data for resetting

        self.select_tables()

    def add_missing_types(self, missing_types, num_tables):
        new_rows = []
        for t in missing_types:
            for _ in range(num_tables):
                table_id = str(uuid.uuid4())
                new_rows.append({'table_id': table_id, 'class': t, 'col_idx': 0, 'data': None})
        df_new = pd.DataFrame(new_rows)
        self.data = pd.concat([self.data, df_new])

    def select_tables(self):
        # Randomly select a portion of tables
        selected_table_ids = random.sample(self.table_ids, int(self.header_ratio * len(self.table_ids)))
        self.data = self.original_data[self.original_data['table_id'].isin(selected_table_ids)]
        self.data = self.data.reset_index(drop=True)
        self.set_data()

    def set_data(self):
        self.process_table_column_wise()


    def process_table_column_wise(self):
        # Create or update the 'sampled_data' column with random samples from 'data'
        self.data['sampled_data'] = self.data['data'].apply(lambda x: random.choices(x, k=self.row_examples))

        # Initialize or clear the "prompt" column
        self.data['prompt'] = ""

        grouped = self.data.groupby('table_id')

        # Construct table-wise prompts
        for table_id, group in grouped:
            base_prompt = ""

            for idx, row in group.iterrows():
                # Use the sampled data for the base prompt construction
                base_prompt += f"Column {row['col_idx']}: {', '.join(row['sampled_data'])}. \n"

            base_prompt = base_prompt.rstrip(" \n")
            base_prompt = f"{self.init_instruction} \n{base_prompt}. \n{self.post_instruction} "

            for idx, row in group.iterrows():
                # Use the sampled data for the final prompt construction
                final_prompt = f"{base_prompt} \nTarget Column: {', '.join(row['sampled_data'])} \nSemantic Type: "
                tokenized_prompt = self.tokenizer(final_prompt, truncation=False, return_tensors='pt')
                tokenized_length = tokenized_prompt.input_ids.shape[1]

                # Ensure the prompt length is within max_length
                if tokenized_length > self.max_length:
                    print('Error', tokenized_length)

                # Correctly access the index of the row in the original DataFrame
                original_idx = row.name  # row.name gives the correct index in self.data
                self.data.at[original_idx, 'prompt'] = final_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        column = self.data.iloc[idx]
        target = column['class']
        input = column['prompt']

        model_inputs = self.tokenizer(input)
        labels = self.tokenizer(target)
        sample_input_ids = model_inputs["input_ids"]
        label_input_ids = labels["input_ids"] + [self.tokenizer.pad_token_id]
        model_inputs["input_ids"] = sample_input_ids + label_input_ids
        labels["input_ids"] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
        sample_input_ids = model_inputs["input_ids"]
        label_input_ids = labels["input_ids"]
        model_inputs["input_ids"] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"]
        labels["input_ids"] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"][:self.max_length])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"][:self.max_length])
        labels["input_ids"] = torch.tensor(labels["input_ids"][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

