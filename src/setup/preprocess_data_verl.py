import argparse
import csv
from copy import deepcopy
from pathlib import Path
import pandas as pd
from datasets import Dataset


def main(data_paths: list[Path], prompt_file: Path, output_folder: Path):
    with open(prompt_file, 'r') as p_f:
        prompt = list(csv.DictReader(p_f))

    data_paths = find_and_identify_splits(data_paths)
    for split, data_path in data_paths.items():
        dataset = Dataset.from_pandas(pd.read_parquet(str(data_path)))

        # split the dataset into train and test
        # train_test_split = dataset.train_test_split(test_size=0.1)
        #
        # train_dataset = train_test_split["train"]
        # test_dataset = train_test_split["test"]

        # add a row to each data item that represents a unique id

        processed_dataset = dataset.map(function=make_map_fn(split, prompt), with_indices=True)
        processed_dataset.to_parquet(output_folder / (data_path.stem + '_' + prompt_file.stem + '.parquet'))


def find_and_identify_splits(dataset_paths: list[Path] = None) -> dict[str, Path]:
    files = []
    for path in dataset_paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(path.glob('*.parquet'))

    data_splits = {}
    for file in files:
        for split in ['test', 'val', 'train']:
            if split in file.stem:
                data_splits[split] = file
    return data_splits


def make_map_fn(split, prompt):
    def process_fn(example, idx):
        note = example["text"]
        prompt_copy = deepcopy(prompt)
        prompt_copy[1]['content'] = prompt_copy[1]['content'].format(note=note)
        target = example["first_code"]
        data = {
            "data_source": "mimic",
            "prompt": prompt_copy,
            "ability": "medicine",
            "reward_model": {
                "style": "rule",
                "ground_truth": target,
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'note':note
                
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', type=Path, default=[Path('../../data')],
                        help='Can be paths to folders with parquet files or paths to parquet files themselves, will '
                             'only use files with train, val or test in the name')
    parser.add_argument('-o', '--output_folder', type=Path, default=Path('../../processed'))
    parser.add_argument('-p', '--prompt_file', type=Path, required=True,
                        help='csv file for the prompts (each row being one turn of dialogue)')

    args = parser.parse_args()

    main(**vars(args))
