import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

import trl
from trl import (
    ModelConfig,
    SFTConfig, 
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from easyalign.utils.load_model_or_tokenizer import load_model, load_tokenizer
from easyalign.utils.build_datasets import build_sft_dataset
from easyalign.utils.utils import setup_logging, init_wandb_training


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    args for callbacks, benchmarks etc
    """
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)
    ###############
    # Setup logging
    ###############
    logger = setup_logging(script_args, training_args, model_args)
    if "wandb" in training_args.report_to:
        init_wandb_training(wandb_project='easyalign-sft')
    ################
    # Load tokenizer
    ################
    logger.info("*** Initializing tokenizer kwargs ***")
    tokenizer = load_tokenizer(model_args.model_name_or_path, padding_side='right')
    ################
    # Load datasets
    ################
    logger.info("*** Initializing dataset kwargs ***")
    train_dataset, eval_dataset = build_sft_dataset(tokenizer, script_args.dataset_name, training_args.max_seq_length)
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    model = load_model(tokenizer , model_args, training_args,  AutoModelForCausalLM)
    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class = tokenizer, 
        peft_config=get_peft_config(model_args)
    )
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
