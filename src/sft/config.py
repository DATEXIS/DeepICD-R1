from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    """
    Additional arguments for training
    """

    model_name: str = field(
        metadata={"help": "The name or path of the pre-trained model to use."}
    )
    dataset_path: str = field(
        metadata={"help": "The path to the dataset file or directory."}
    )
