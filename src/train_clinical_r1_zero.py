import logging
import os
from datetime import datetime

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from dataclasses import dataclass
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig,GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import pandas as pd
from rewards import trl_batched_compute_score_multiple_think_traces, trl_batched_length_malus_score, LENGTH_MALUS, trl_batched_traces_llm_score
import wandb
########################
# Custom dataclasses
########################
########################
@dataclass
class ScriptArguments:
    train_data: str
    eval_data: str
    gpu_memory_utilization: float
    use_llm_as_a_judge: bool
    tokenizer_name_or_path: str


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token




    


    reward_funcs = [trl_batched_compute_score_multiple_think_traces]
    if LENGTH_MALUS != 0:
        reward_funcs.append(trl_batched_length_malus_score)
    if script_args.use_llm_as_a_judge:
        reward_funcs.append(trl_batched_traces_llm_score)

    # convert our dataset to the r1 prompt
    """
    dataset = dataset.map(lambda x: generate_r1_prompt(x["TEXT"], x["LONG_CODES"]))
    dataset = dataset.select_columns(["prompt", "target", "note"])


    # split the dataset into train and test

    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    """
    train_dataset = Dataset.from_parquet(script_args.train_data)
    eval_dataset = Dataset.from_parquet(script_args.eval_data)
    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_state()
    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    """
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()
    """
    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
