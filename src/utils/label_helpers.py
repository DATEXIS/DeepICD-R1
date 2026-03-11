import csv
import json
import logging
from enum import IntEnum, Enum
from pathlib import Path
from typing import Iterable, Callable
import pickle
import torch
from meilisearch import Client
from meilisearch.models.document import Document

from utils import log_function_call

logger = logging.getLogger(__name__)


class ICDVersion(IntEnum):
    ICD9 = 9
    ICD10 = 10


class ICD10Chapter(Enum):
    I = ("A00", "C00")
    II = ("C00", "D50")
    III = ("D50", "E00")
    IV = ("E00", "F00")
    V = ("F00", "G00")
    VI = ("G00", "H00")
    VII = ("H00", "H60")
    VIII = ("H60", "I00")
    IX = ("I00", "J00")
    X = ("J00", "K00")
    XI = ("K00", "L00")
    XII = ("L00", "M00")
    XIII = ("M00", "N00")
    XIV = ("N00", "O00")
    XV = ("O00", "P00")
    XVI = ("P00", "Q00")
    XVII = ("Q00", "R00")
    XVIII = ("R00", "S00")
    XIX = ("S00", "U00")
    XXII = ("U00", "V00")  # special purposes, so out of order for some reason
    XX = ("V00", "Z00")
    XXI = ("Z00", "a00")

    def __init__(self, start: str, end: str):
        self.start = start
        self.end = end

    @classmethod
    def from_file(cls, file: Path, icd_version: int, is_pickle: bool):
        if is_pickle:
            with open(file, "rb") as f:
                labels = pickle.load(f)
        else:   
            with open(file) as f:
                labels = json.load(f)
        return cls(labels, icd_version)
    def __len__(self):
        return len(self.label2id)
        
    
class FirstDiaVocabulary:
    def __init__(self, labels: Iterable[str], icd_version: int, truncate: bool = False):
        self.label2id = {}

        self.icd_version = icd_version
        if truncate:
            self.truncate_fn = truncate_labels_9 if icd_version == 9 else truncate_labels_10
            labels = self.truncate_fn(labels)
        else:
            self.truncate_fn = None

        self.add_additional_labels(labels)  # we don't know whether labels have duplicates

    def add_additional_labels(self, labels):
        if self.truncate_fn:
            labels = self.truncate_fn(labels)
        for label in labels:
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)

    @classmethod
    def from_file(cls, file: Path, icd_version: int, is_pickle: bool):
        if is_pickle:
            with open(file, "rb") as f:
                labels = pickle.load(f)
        else:   
            with open(file) as f:
                labels = json.load(f)
        return cls(labels, icd_version)

    def __len__(self):
        return len(self.label2id)

class Vocabulary:
    truncate_fn: Callable[[Iterable[str]], list[str]]

    def __init__(self, labels: Iterable[str], icd_version: int, truncate: bool = False, to_chapter: bool = False):
        self._full2id = {}
        self._label2id = {}
    
        self.icd_version = icd_version

        if truncate and to_chapter:
            raise ValueError(
                'You can only either truncate or translate to chapter; build 2 Vocabularies if you want both')

        if truncate:
            self.truncate_fn = truncate_labels_9 if icd_version == 9 else truncate_labels_10
        elif to_chapter:
            if icd_version == 9:
                raise NotImplementedError
            self.truncate_fn = get_icd10_chapters
        else:
            self.truncate_fn = lambda x: x

        self.add_additional_labels(labels)  # we don't know whether labels have duplicates

    @classmethod
    @log_function_call(logger=logger)
    def from_file(cls, file: Path, icd_version: int):
        if file.suffix == '.json':
            with open(file) as f:
                labels = json.load(f)
        elif file.suffix == '.csv':
            with open(file, newline='') as f:
                csv_content = list(csv.reader(f))
                codes_col = csv_content[0].index('icd_codes')
                labels = [cols[codes_col] for cols in csv_content[1:]]
        else:
            raise ValueError(f'Unsupported file type: {file.suffix}')
        return cls(labels, icd_version)

    def add_additional_labels(self, labels):
        trunc_labels = self.truncate_fn(labels)

        for label, trunc_label in zip(labels, trunc_labels):
            if trunc_label not in self._full2id:
                label_id = len(self._label2id)
                self._label2id[trunc_label] = label_id
                self._full2id[trunc_label] = label_id
            if label not in self._full2id:
                self._full2id[label] = self._label2id[trunc_label]


    @property
    def labels(self) -> list[str]:
        return list(self._label2id.keys())

    def translate_batched_labels(self, batch: Iterable[Iterable[str]]):
        _ = []
        for labels in batch:
            for label in labels:
                if label in self.labels:
                    _.append(self[label])
                else:
                    _.append(self["NIL"])
        return _

    def safe_translate_batched_labels(self, batch: Iterable[Iterable[str]]):
        return [[self[label] for label in labels if label in self._full2id] for labels in batch]

    def __getitem__(self, item):
        return self._full2id[item]

    def __len__(self):
        return len(self._label2id)


def truncate_labels_9(labels: Iterable[str]) -> list[str]:
    return [truncate_label_9(label) for label in labels]


def truncate_label_9(label: str) -> str:
    return label[:4] if label.startswith('E') else label[:3]


def truncate_labels_10(labels: Iterable[str]) -> list[str]:
    return [truncate_label_10(label) for label in labels]


def truncate_label_10(label: str) -> str:
    return label[:3]


def get_icd10_chapters(codes: Iterable[str]) -> list[str]:
    return [get_icd10_chapter(code).name for code in codes]


def get_icd10_chapter(code: str) -> ICD10Chapter | None:
    code = code.upper()

    for chapter in ICD10Chapter:
        if code < chapter.end:
            return chapter
    return None


def get_all_icd_codes_from_meili(meili_client: Client, icd_code_index: str) -> list[str]:
    all_results = _retrieve_all_from_meili(meili_client, icd_code_index)
    return [result.icd_code.strip() for result in all_results]


def get_all_icd_information(meili_client: Client, icd_version: ICDVersion) -> dict[str, Document]:
    all_results = _retrieve_all_from_meili(meili_client, f'icd_{"9" if icd_version == 9 else "10"}_descriptions')
    return {r.icd_code.strip(): r for r in all_results}


def _retrieve_all_from_meili(meili_client: Client, icd_code_index: str) -> list[Document]:
    index = meili_client.index(icd_code_index)
    limit = 10_000
    offset = 0
    all_results = []
    while True:
        docs = index.get_documents({'limit': limit, 'offset': offset})
        if not docs.results:
            break
        all_results.extend(docs.results)
        offset += limit
    return all_results


def to_multi_hot(batched_indices, label2id: Vocabulary) -> torch.Tensor:
    multi_hot = torch.zeros(len(batched_indices), len(label2id), dtype=torch.int32)
    for i, ids in enumerate(batched_indices):
        multi_hot[i, ids] = 1
    return multi_hot


def to_fake_logits(batched_indices, label2id: Vocabulary) -> torch.Tensor:
    """Acts as if the order of the labels represents some sort of certainty that can be turned into logits, for top-k
    main diagnosis analysis"""
    fake_logits = torch.zeros(len(batched_indices), len(label2id), dtype=torch.float32)
    for i, ids in enumerate(batched_indices):
        fake_logits[i, ids] = 1 / (i + 2) + 0.5
    return fake_logits


def load_labels(file):
    with open(file) as f:
        labels = json.load(f)

    label2id = {label: id for id, label in enumerate(labels, 1)}
    return label2id
