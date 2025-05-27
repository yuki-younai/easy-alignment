import torch
from torch.utils.data import Dataset, DataLoader
from datasets import  load_dataset, load_from_disk
from typing import Dict, Optional, Sequence
import random


def build_sft_dataset(tokenizer, data_path, max_seq_length) -> Dataset:

    train_dataset = load_dataset(split="train", path=data_path)
    train_dataset = train_dataset.shuffle()

    #Bespoke-Stratos-17k
    def split_prompt_and_responses_Bespoke(examples):
        messages = []
        for conv in examples['conversations']:
            if conv['from']=='user':
                messages.append({"role":'user', "content":conv['value']})
            else:
                messages.append({"role":'assistant', "content":conv['value']})

        return {
            "messages": messages
        }
    def split_prompt_and_responses_alpaca(example) -> Dict[str, str]:
        prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example['instruction']+ example['input']}],
                tokenize=False,
                add_generation_prompt=True,
            )
        completion =  example["output"] + tokenizer.eos_token + '\n'
        return {"prompt": prompt,
                "completion": completion}
    
    if 'alpaca' in data_path:
        train_dataset = train_dataset.map(split_prompt_and_responses_alpaca, remove_columns=train_dataset.column_names, num_proc=8)
    else:
        ValueError()
    
    return train_dataset, None


def build_dpo_dataset(tokenizer, data_path, max_seq_length, task_type="helpful") -> Dataset:

    dataset = load_dataset(split="train", path=data_path)

    def split_prompt_and_responses_hhrlhf(sample) -> Dict[str, str]:
        search_term = "\n\nAssistant:"
        search_term_idx = sample["chosen"].rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        prompt = sample["chosen"][: search_term_idx + len(search_term)] + tokenizer.eos_token
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :] + tokenizer.eos_token,
            "rejected": sample["rejected"][len(prompt) :] + tokenizer.eos_token,
        }
    
    def split_prompt_and_responses_ultrafeedback(example):
        prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example['prompt']}],
                tokenize=False,
                add_generation_prompt=True,
            )
        chosen = example['chosen'][1]['content'] + tokenizer.eos_token
        rejected = example['rejected'][1]['content'] + tokenizer.eos_token
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    if 'hh-rlhf' in data_path:
        return dataset.map(split_prompt_and_responses_hhrlhf, num_proc=8)
    if 'ultrafeedback' in data_path:
        return dataset.map(split_prompt_and_responses_ultrafeedback, num_proc=8)


def build_kto_dataset(tokenizer, data_path, max_seq_length) -> Dataset:

    train_dataset = load_dataset(split="train", path=data_path)

    def split_prompt_and_responses_ultrafeedback(example) -> Dict[str, str]:
        prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example['prompt']}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if random.random()<0.5:
            completion = example['chosen'][1]['content']+ tokenizer.eos_token
            label = True
        else:
            completion = example["rejected"][1]['content']+ tokenizer.eos_token
            label = False
        return {
            "prompt": prompt,
            "completion": completion,
            "label": label
        }
    
    def split_prompt_and_responses_ktomix(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"][-max_seq_length//2:], tokenize=False, add_generation_prompt=True)
        example["completion"] = tokenizer.apply_chat_template(example["completion"][:max_seq_length//2], tokenize=False)
        return example

    if 'kto' in data_path:
        return train_dataset.map(split_prompt_and_responses_ktomix,  num_proc=8)
    if 'ultrafeedback' in data_path:
        return train_dataset.map(split_prompt_and_responses_ultrafeedback,  remove_columns=train_dataset.column_names, num_proc=8)
    
    
def build_grpo_dataset(tokenizer, data_path, max_seq_length, template='template') -> Dataset:

    #train_dataset = load_dataset(split="train", path=data_path)
    train_dataset = load_from_disk(data_path)
    #eval_dataset = load_dataset(split="test", path=data_path)
    train_dataset = train_dataset.shuffle()

    SYSTEM_PROMPT = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    )
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example['solution']
        }
    # Format into conversation
    def make_r1_conversation(example):
        return {
            "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + example["problem"]
        + "\nAssistant: <think>",
            "solution": example['solution']
        }
    if template=='template':
        train_dataset = train_dataset.map(make_conversation, num_proc=8)
    elif template=='r1':
        train_dataset = train_dataset.map(make_r1_conversation, num_proc=8)
        
    #eval_dataset = eval_dataset.map(make_conversation,  num_proc=8)

    return train_dataset, None

















