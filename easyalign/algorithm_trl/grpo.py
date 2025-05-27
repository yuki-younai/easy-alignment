from dataclasses import dataclass, field
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, GRPOConfig, GRPOTrainer
#from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, GRPOConfig
from typing import Optional
from easyalign.reward.openr1_reward import  (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward, 
    understand_reward
)

from easyalign.utils.load_model_or_tokenizer import load_model, load_tokenizer
from easyalign.utils.build_datasets import build_grpo_dataset
from easyalign.utils.utils import setup_logging, init_wandb_training


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    template: str = field(
        default='template',
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )

def main():
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # training_args.loss_type = script_args.loss_type
    # training_args.cliprange = script_args.cliprange
    set_seed(training_args.seed)
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
    tokenizer = load_tokenizer(model_args.model_name_or_path, padding_side='right')
    ################
    # Load datasets
    ################
    logger.info("*** Initializing dataset kwargs ***")
    train_dataset, eval_dataset = build_grpo_dataset(tokenizer, script_args.dataset_name, training_args.max_prompt_length, template=script_args.template)
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    model = load_model(tokenizer , model_args, training_args,  AutoModelForCausalLM)
    ###################
    # Reward Model init kwargs
    ###################
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "tag_count": tag_count_reward,
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "undertand_reward": understand_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model= model,
        processing_class = tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config= get_peft_config(model_args)
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
    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
