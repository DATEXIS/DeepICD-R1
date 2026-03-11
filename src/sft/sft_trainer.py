import json
from typing import Optional

from datasets import Dataset

from trl import SFTTrainer, SFTConfig, TrlParser

from config import TrainingArguments


class MyTrainer:

    def __init__(self, sft_trainer_config: SFTConfig, training_arguments: TrainingArguments):
        self.sft_trainer_config = sft_trainer_config
        self.model_name = training_arguments.model_name
        self.dataset_path = training_arguments.dataset_path

    def train(self):
        train_data = Dataset.from_parquet(self.dataset_path)

        trainer = SFTTrainer(
            model=self.model_name,
            train_dataset=train_data,
            args=self.sft_trainer_config,
        )

        trainer.train()

    def _load_dataset(self) -> Dataset:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        return Dataset.from_dict(data_json)

if __name__ == '__main__':
    parser = TrlParser((SFTConfig, TrainingArguments))
    sft_trainer_config, training_arguments = parser.parse_args_and_config()
    my_trainer = MyTrainer(sft_trainer_config=sft_trainer_config, training_arguments=training_arguments)
    my_trainer.train()
    
