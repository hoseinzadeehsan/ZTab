import openai
import os
import configargparse



parser = configargparse.ArgParser()
# Configuration

parser.add_argument('--batch', type=int, default=32,
                    help="Batch size used for training, validation, and testing.")
parser.add_argument('--data-name', type=str, default='dataset-sota',
                    help="Dataset name to be loaded for training and evaluation.")
parser.add_argument('--model-name', type=str, default='gpt-4o-mini',
                    help="Base model used for fine-tuning.")
parser.add_argument("--epoch", type=int, default=1,
                    help="Total number of training epochs.")
parser.add_argument('--token', type=str, default='',
                    help="Hugginface access token.")

args = parser.parse_args()
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# Set your OpenAI API key (replace with your actual key or set as environment variable)
os.environ["OPENAI_API_KEY"] = args.token
client = openai.OpenAI()

if args.model_name == 'gpt-3.5':
    model_name = "gpt-3.5-turbo-0125"
elif args.model_name == 'gpt-4o-mini':
    model_name = "gpt-4o-mini-2024-07-18"
elif args.model_name == 'gpt-4o':
    model_name = "gpt-4o-2024-08-06"
elif args.model_name == 'gpt-4.1-mini':
    model_name = "gpt-4.1-mini-2025-04-14"


if args.data_name == 'dataset-sota-small':
    jsonl_file = "sota-schema-small.jsonl"
elif args.data_name == 'dataset-sota':
    jsonl_file = "sota-schema.jsonl"
elif args.data_name == 'dataset-sota-dbpedia':
    jsonl_file = "sota-dbpedia.jsonl"
elif args.data_name == 'dataset-t2d':
    jsonl_file = "t2d.jsonl"
elif args.data_name == 'dataset-limaye':
    jsonl_file = "limaye.jsonl"
elif args.data_name == 'dataset-wikipedia':
    jsonl_file = "wikipedia.jsonl"
elif args.data_name == 'dataset-turl':
    jsonl_file = "turl.jsonl"
else:
    raise ValueError(f"Unsupported dataset name: {args.data_name}")

jsonl_path = os.path.join(data_dir, jsonl_file)
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")


# Hyperparameters for fine-tuning
hyperparams = {
    "n_epochs": args.epoch,  # Number of epochs (default 3, ZTab uses 20 for larger datasets)
    "batch_size": args.batch #"auto"  # Default, or set to an integer like 4 or 8
}

print(f"Processing file: {jsonl_path}")

# Step 1: Upload the JSONL file to OpenAI
with open(jsonl_path, 'rb') as f:
    file_response = client.files.create(
        file=f,
        purpose='fine-tune'
    )
file_id = file_response.id
print(f"Uploaded file ID: {file_id}")

# model_name = "gpt-4o-mini-2024-07-18"
fine_tune_response = client.fine_tuning.jobs.create(
    training_file=file_id,
    model=model_name,
    hyperparameters=hyperparams
)
job_id = fine_tune_response.id
print(f"Fine-tuning job created for {jsonl_path}: {job_id}")
