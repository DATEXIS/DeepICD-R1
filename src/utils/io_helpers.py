import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Generator

import torch
from datasets import concatenate_datasets, Dataset

from utils import log_function_call

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    data: Path
    output_path: Path = Path("./output/")
    start_row: int = 0
    end_row: int = None


@log_function_call(logger=logger)
def check_and_create_output_path(output_path: Path, default_file_name: str | Path, start_row: int = 0,
                                 end_row: int = None) -> Path:
    if end_row:
        idx_range = f'_{start_row}-{end_row}'
    elif start_row:
        idx_range = f'_from-{start_row}'
    else:
        idx_range = ''

    if output_path.suffix == '':
        output_file = output_path / (default_file_name + idx_range)
    else:
        output_file = output_path.with_stem(output_path.stem + idx_range)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    return output_file


@log_function_call(logger=logger)
def read_dataset(path: Path, read_columns: list[str], start: int = 0, end: int = None) -> Dataset:
    if path.suffix == '.parquet':
        dataset = Dataset.from_parquet(str(path), columns=read_columns)
    else:
        paths = list(path.glob('**/*.parquet'))
        logger.info('Detected %s', paths)
        datasets = [Dataset.from_parquet(str(path), columns=read_columns) for path in paths]

        if len(datasets) == 0:
            raise FileNotFoundError(f"No .parquet datasets found at {path}")

        dataset = concatenate_datasets(datasets)

    if end is None:
        end = len(dataset)

    return dataset.select(range(start, end))


@log_function_call(logger=logger)
def read_negative_samples(path: Path):
    try:
        if path.suffix == '':
            return Dataset.from_json(str(path / 'negative_samples.jsonl'))
        else:
            return Dataset.from_json(str(path.parent / 'negative_samples.jsonl'))
    except FileNotFoundError:
        logger.warning(f"Could not find negative_samples.jsonl for {path}, continuing without it.")
        return Dataset.from_dict(dict(hadm_id=[], negatives=[]))


@log_function_call(logger=logger)
def save_dataset(dataset: Dataset, output_file: Path) -> None:
    if output_file.suffix == '.parquet':
        dataset.to_parquet(output_file)
    else:
        # keeps any other suffixes
        dataset.to_parquet(output_file.with_name(f'{output_file.name}.parquet'))

@contextmanager
def temp_checkpoint_file_from_target_file(output_file: Path) -> Generator[Path, Any, None]:
    checkpoint_path = output_file.with_name(f'{output_file.name}_ckpt.jsonl')
    yield checkpoint_path

    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass

def get_processed_indices(file_path: str, index_name:str = '_idx') -> set[int]:
    """Return the set of _idx already processed in checkpoint."""
    processed = set()
    if not os.path.exists(file_path):
        return processed
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                processed.add(row[index_name])
            except JSONDecodeError:
                continue  # skip corrupt lines
    if len(processed) != 0:
        logger.info(f'Recovered {len(processed)} rows from {file_path}.')
    return processed

def to_serializable(obj):
    # catch tensor since they're not serializable
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj