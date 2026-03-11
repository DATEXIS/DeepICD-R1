import logging
from typing import Any

import torch
import wandb
from torch import tensor, Tensor
from torchmetrics import MetricCollection, Recall, Precision, F1Score, Accuracy
from torchmetrics.classification import MultilabelStatScores
from torchmetrics.retrieval import RetrievalAUROC
from wandb import Table

logger = logging.getLogger(__name__)

top_ks = [1, 3, 5, 10, 20]


class MultilabelSkipAUROC(RetrievalAUROC):
    def __init__(self, num_labels):
        super().__init__(empty_target_action='skip')
        self.register_buffer('class_indices', torch.arange(0, num_labels))

    def update(self, preds: Tensor, target: Tensor, **kwargs) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators.
        """
        return super().update(preds, target, self.class_indices.expand(preds.shape[0], -1))


def create_metrics(num_classes, prefix: str = None):
    return MetricCollection(
        micro_and_macro(Recall, 'Recall', 'multilabel', num_labels=num_classes) |
        micro_and_macro(Precision, 'Precision', 'multilabel', num_labels=num_classes) |
        micro_and_macro(F1Score, 'F1', 'multilabel', num_labels=num_classes) |
        micro_and_macro(Accuracy, 'Accuracy', 'multilabel', num_labels=num_classes) |

        {  # 'AUROC': MultilabelSkipAUROC(num_labels=num_classes),
            'MacroStats': MultilabelStatScores(num_labels=num_classes, average='macro'),
            'MicroStats': MultilabelStatScores(num_labels=num_classes, average='micro')},
        prefix=prefix,
    )


def create_main_diagnosis_metrics(num_classes, prefix: str = None, postfix: str = None):
    return MetricCollection(
        micro_and_macro(Recall, 'Recall', 'multiclass', num_classes=num_classes, at_x=False) |
        micro_and_macro(Precision, 'Precision', 'multiclass', num_classes=num_classes, at_x=False) |
        micro_and_macro(F1Score, 'F1', 'multiclass', num_classes=num_classes, at_x=False) |
        micro_and_macro(Accuracy, 'Accuracy', 'multiclass', num_classes=num_classes, at_x=False),
        prefix=prefix,
        postfix=postfix
    )

def create_main_diagnosis_fake_accuracy(num_classes, prefix: str = None, postfix: str = None):
    return MetricCollection(
        micro_and_macro(Recall, 'Recall', 'multiclass', num_classes=num_classes, at_x=True),
        prefix=prefix,
        postfix=postfix
    )

def micro_and_macro(metric_cls, name, *args, at_x=False,
                    prefix_no_avg='NoAvg', prefix_micro='Micro', prefix_macro='Macro',
                    **kwargs):
    if not at_x:
        metrics = {prefix_no_avg + name: metric_cls(*args, **kwargs, average=None),
                   prefix_micro + name: metric_cls(*args, **kwargs, average='micro'),
                   prefix_macro + name: metric_cls(*args, **kwargs, average='macro')}
    else:
        metrics = build_metric_at_x(metric_cls, prefix_micro + name, *args, **kwargs, average='micro')
        metrics |= build_metric_at_x(metric_cls, prefix_macro + name, *args, **kwargs, average='macro')

    return metrics


def build_metric_at_x(metric_cls, name, *args, add_none: bool = False, **kwargs):
    return (({f'{name}@N': metric_cls(*args, **kwargs, top_k=None)} if add_none else {})
            | {f'{name}@{k}': metric_cls(*args, **kwargs, top_k=k) for k in top_ks})


def log_all_collections(*metric_collections: MetricCollection, labels: list[str]):
    single_result_metrics_dict = {}
    third_of_labels = len(labels) // 3
    print("LOG ALL START")
    for metric_collection in metric_collections:
        print(f"ALL METRICS: {metric_collection}")
        computed_collection = metric_collection.compute()
        for metric_name, metric_result in computed_collection.items():
            print(f"METRIC NAME: {metric_name}")
            print(f"METRIC RESULT: {metric_result}")
            if metric_result.numel() < 2:
                single_result_metrics_dict[metric_name] = metric_result
            elif metric_result.shape[0] > 5:
                single_result_metrics_dict[metric_name.replace('NoAvg', 'Top')] = metric_result[:third_of_labels].mean()
                single_result_metrics_dict[metric_name.replace('NoAvg', 'Mid')] = metric_result[
                    third_of_labels:third_of_labels * 2].mean()
                single_result_metrics_dict[metric_name.replace('NoAvg', 'Bot')] = metric_result[
                    third_of_labels * 2:].mean()
                # log_multi_element_tensor_as_table(metric_name, metric_result, labels)
            else:
                # log_multi_element_tensor_as_table(metric_name, metric_result, ['TP', 'FP', 'TN', 'FN', 'SUP'])
                pass

    wandb.log(single_result_metrics_dict, commit=False)


def get_non_averaged_results(*metric_collections: MetricCollection, labels: list[str]):
    non_averaged_results = {'NoAvg': {'columns': labels}, 'conf': {'columns': ['TP', 'FP', 'TN', 'FN', 'SUP']}}

    for metric_collection in metric_collections:
        computed_collection = metric_collection.compute()
        for metric_name, metric_result in computed_collection.items():
            if metric_result.numel() < 2:
                pass
            elif metric_result.shape[0] == len(labels):
                non_averaged_results['NoAvg'][metric_name] = metric_result
            else:
                non_averaged_results['conf'][metric_name] = metric_result
    return non_averaged_results


def log_multi_element_tensor_as_table(metric_name: str, metric_result: tensor, columns: list[str], commit=False):
    shape = metric_result.shape
    print("HIT LOG TENSOR")
    if shape[0] != len(columns):
        raise NotImplementedError(f'Can\'t match metric shape {metric_result.shape} to columns {columns}!')

    if len(shape) == 1:
        print("1")
        table_data = [metric_result.tolist()]
        print("1 fertig")
    elif shape[1] == len(columns):
        print("2")
        table_data = [[column_name] + row for column_name, row in zip(columns, metric_result.tolist())]
        columns = ['id'] + columns
    else:
        print("3")
        table_data = metric_result.tolist()
    wandb.log({metric_name: Table(columns=columns, data=table_data, allow_mixed_types=True)}, commit=commit)


def log_dict_as_table(name: str, result_dict: dict[str, list[Any]], commit: bool = True):
    """
    Args:
        name: name of the table
        result_dict: dict with various results, which will be turned into columns
    """
    other_columns = list(result_dict.keys())
    row_oriented_data = [list(batched_result) for batched_result in zip(*result_dict.values())]

    table = Table(columns=other_columns, data=row_oriented_data)
    wandb.log({name: table}, commit=commit)
