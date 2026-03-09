from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

def load_model_tokenizer_and_peft_config(args, access_token, device, quantized=False):
    if args.model_name == 'mistral':
        model_name = "mistralai/Mistral-7B-v0.1"
    elif args.model_name == 'mistral-instruct':
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif args.model_name == 'mixtral':
        model_name = 'mistralai/Mixtral-8x7B-v0.1'
    elif args.model_name == 'mixtral-instruct':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    elif args.model_name == 'llama3':
        model_name = "meta-llama/Meta-Llama-3-8B"
    elif args.model_name == 'llama3-instruct':
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model_name == 'llama3.1':
        model_name = "meta-llama/Meta-Llama-3.1-8B"
    elif args.model_name == 'llama3.1-instruct':
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.model_name == 'phi3':
        # model_name = "microsoft/Phi-3-small-128k-instruct"
        model_name = "microsoft/Phi-3-mini-4k-instruct"
    elif args.model_name == 'Qwen7':
        model_name = "Qwen/Qwen2.5-7B"
    elif args.model_name == 'Qwen8':
        model_name = "Qwen/Qwen3-8B"
    elif args.model_name == 'Qwen1.5':
        model_name = "Qwen/Qwen2.5-1.5B"
    elif args.model_name == 'Qwen14':
        model_name = "Qwen/Qwen2.5-14B"

    if quantized:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            device_map=device,
            quantization_config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

    else:
        # For non-quantized models
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            device_map=device,
            trust_remote_code=True
        )

    if args.model_name == 'phi3':
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side='left',
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    initial_model = model
    return model, tokenizer, initial_model

