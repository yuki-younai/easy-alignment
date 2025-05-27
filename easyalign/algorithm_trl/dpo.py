from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, HfArgumentParser
from trl import ModelConfig, ScriptArguments, TrlParser, create_reference_model, get_peft_config, DPOConfig, DPOTrainer
from transformers import HfArgumentParser, AutoModelForCausalLM

from easyalign.utils.load_model_or_tokenizer import load_model, load_tokenizer
from easyalign.utils.build_datasets import build_dpo_dataset
from easyalign.utils.utils import setup_logging, init_wandb_training

@dataclass
class DPOScriptArguments:
    dataset_name: str = "trl-lib/kto-mix-14k"
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    task_type: str = "helpful"

def train():
    parser = HfArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    set_seed(seed=42)
    ###############
    # Setup logging
    ###############
    logger = setup_logging(script_args, training_args, model_args)
    if "wandb" in training_args.report_to:
        init_wandb_training(wandb_project=script_args.wandb_project)
    ################
    # Load tokenizer
    ################
    logger.info("*** Initializing tokenizer kwargs ***")
    tokenizer = load_tokenizer(model_args.model_name_or_path)
    ################
    # Load datasets
    ################
    logger.info("*** Initializing dataset kwargs ***")
    train_dataset = build_dpo_dataset(tokenizer, script_args.dataset_name, training_args.max_length)
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    model = load_model(tokenizer, model_args, training_args,  AutoModelForCausalLM)
    ref_model = None
    ################
    # Training
    ################
    logger.info("*** Train ***")
    dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    peft_config=get_peft_config(model_args)
                )

    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()


if __name__ == "__main__":
    train()
