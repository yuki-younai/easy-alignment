import datasets
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, HfArgumentParser, AutoModelForSequenceClassification
from dataclasses import dataclass, field
from typing import Optional

import trl
from trl import (
    ModelConfig,
    PPOConfig, 
    PPOTrainer,
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


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
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
    tokenizer = load_tokenizer(model_args.model_name_or_path, padding_side='left')
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    policy = load_model(tokenizer , model_args, training_args,  AutoModelForCausalLM)
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = load_model(tokenizer , model_args, training_args,  AutoModelForCausalLM)
    else:
        ref_policy = None

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    ################
    # Dataset
    ################
    logger.info("*** Initializing dataset kwargs ***")
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    #train_dataset = dataset['train']
    train_dataset = dataset['validation']
    eval_dataset = dataset['validation'] if training_args.eval_strategy != "no" else None
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    trainer.generate_completions()
