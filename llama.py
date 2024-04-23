import json
import re
from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
dataset = load_dataset("Salesforce/dialogstudio", "TweetSumm")
dataset

DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()


def generate_training_prompt(
        conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{conversation.strip()}

### Response:
{summary}
""".strip()


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)


def create_conversation_text(data_point):
    text = ""
    for item in data_point["log"]:
        user = clean_text(item["user utterance"])
        text += f"user: {user.strip()}\n"

        agent = clean_text(item["system response"])
        text += f"agent: {agent.strip()}\n"

    return text


#Most important part here is getting the summaries
def generate_text(data_point):
    summaries = json.loads(data_point["original dialog info"])["summaries"][
        "abstractive_summaries"
    ]
    summary = summaries[0]
    summary = " ".join(summary)

    conversation_text = create_conversation_text(data_point)
    return {
        "conversation": conversation_text,
        "summary": summary,
        "text": generate_training_prompt(conversation_text, summary),
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_text)
        .remove_columns(
            [
                "original dialog id",
                "new dialog id",
                "dialog index",
                "original dialog info",
                "log",
                "prompt",
            ]
        )
    )

dataset["train"] = process_dataset(dataset["train"])
dataset["validation"] = process_dataset(dataset["validation"])
dataset

notebook_login()
