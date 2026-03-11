import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, astuple
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
from transformers import AutoTokenizer
import datasets
import plotly.express as px
import torch
import wandb
from datasets import Dataset, tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelRecall, MulticlassConfusionMatrix
from transformers import HfArgumentParser
from vllm import LLM
from wandb import Table
from wandb.sdk.data_types import _dtypes
import re
from rewards import OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX, batched_msearch_meilisearch, longest_common_prefix, \
    sort_by_highest_reward, separate_by_first_character_with_duplicates, separate_by_first_character, RETRIEVER_CLIENT, ICD_10_CM_PATTERN
from utils.formatter import format_ground_truth, format_predictions, format_diagnosis, format_match, \
    MatchWithTraceAndQuery, MatchWithTraceAndPrediction
from utils.io_helpers import read_dataset, to_serializable
from utils.label_helpers import to_multi_hot, get_all_icd_codes_from_meili, Vocabulary, ICDVersion, to_fake_logits, get_icd10_chapter
from utils.metrics import create_metrics, log_all_collections, log_dict_as_table, create_main_diagnosis_metrics, \
    get_non_averaged_results, create_main_diagnosis_fake_accuracy

CODE_ONLY_PATTERN =  re.compile(r'\b(?P<code>[A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?)\b', re.I)
########################
# Setup logging
########################
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info('Disabling datasets caching')
datasets.disable_caching()


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    model_name: str
    data: Path
    labels_file: Path = None
    output_dir: Path = Path("output")
    tokenizer_name: str = None
    batch_size: int = 8
    prompt_column: str = "prompt"
    target_column: str = 'icd_codes'
    multiple_think_traces: bool = True
    num_gpus: int = torch.cuda.device_count()
    min_max_tokens: int = 1024
    icd_version: ICDVersion = ICDVersion.ICD10
    dtype: str = 'bfloat16'
    limit_examples: int = None
    log_matches_detail: bool = False
    reevaluate_results: Path = None
    icd_code_index: str = 'mimic_icd_{icd_version}_codes'
    md_index: str = 'mimic_icd_{icd_version}_first_diagnosis_codes'
    code_only: bool = False


@torch.inference_mode()
def main(args: ScriptArguments):
    torch._dynamo.disable()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data: %s", args.data)

    read_columns = ['hadm_id', 'subject_id', args.target_column]
    if not args.reevaluate_results:
        read_columns += [args.prompt_column]

    dataset = read_dataset(args.data, end=args.limit_examples, read_columns=read_columns)
    
    logger.info('Setting up metrics')

    #meilisearch_labels = get_all_icd_codes_from_meili(RETRIEVER_CLIENT,
    #                                                  args.icd_code_index.format(icd_version=args.icd_version))

    if args.labels_file is None:
        args.labels_file = args.data / 'icd_codes.csv' if args.data.is_dir() else args.data.parent / 'icd_codes.csv'
    vocab_f = Vocabulary.from_file(args.labels_file, args.icd_version)

    vocabs = {'full_code': vocab_f,
              '3digit_code': Vocabulary(vocab_f.labels, args.icd_version, truncate=True),
              'chapter': Vocabulary(vocab_f.labels, args.icd_version, to_chapter=True)
              }


    evaluators = [Evaluator(vocab, name) for name, vocab in vocabs.items()]

    llm = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    if not args.reevaluate_results:
        logger.info("Loading model: %s", args.model_name)

        llm = LLM(model=args.model_name,
                  enable_prefix_caching=True,
                  tensor_parallel_size=args.num_gpus,
                  dtype=args.dtype,
                  download_dir=os.getenv('HF_HOME', None)
                  )
        set_llm_to_min_max_tokens_or_default(llm, args.min_max_tokens)
        logger.info(f"Set output length to: {args.min_max_tokens}")
        def filter_long_prompt(example, tokenizer, max_tokens):
            token_ids =  tokenizer(" ".join(msg["content"] for msg in example[args.prompt_column]))
            return len(token_ids) <= max_tokens
        logger.info(f"Filter prompts longer than 2048")
        logger.info(f"Current dataset size: {len(dataset)}")
        dataset = dataset.filter(lambda x: filter_long_prompt(x, llm.get_tokenizer(), 2048))
        logger.info(f"Dataset size after filterin: {len(dataset)}")
        logger.info("Setting up DataLoader")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=device == 'cuda',
                                collate_fn=make_collate_fn(
                                    tokenizer=llm.get_tokenizer(),
                                    prompt_col=args.prompt_column,
                                    target_col=args.target_column,
                                    vocabs=vocabs,
                                ))
        run_config_add_ons = {'sampling_params': llm.get_default_sampling_params() if llm is not None else None}
    else:  # load already completed evaluation results
        if args.reevaluate_results.is_dir():
            parquet_files = list(args.reevaluate_results.glob('*.parquet'))
            if len(parquet_files) != 1:
                raise RuntimeError(f'Too many or too few parquet files (needs to be exactly 1): {parquet_files}')
            args.reevaluate_results = parquet_files[0]

        previous_results = Dataset.from_parquet(str(args.reevaluate_results)).with_format('torch')
        dataloader = BatchedDatasetLoader(previous_results, batch_size=args.batch_size,
                                          answer_col='answers', target_col=args.target_column, vocabs=vocabs)

        dataset = merge_previous_results_to_dataset(dataset, previous_results)
        if len(previous_results) != len(dataset):
            raise RuntimeError(f'Dataset didn\'t contain all entries for reevaluation!'
                               f' {len(dataset)} / {len(previous_results)} matching.')

        run_config_add_ons = {}

    logger.info("Starting evaluation")

    all_completions = []
    all_eval_results = defaultdict(list)
    all_matches = []
    with wandb.init(config=asdict(args) | run_config_add_ons) as run:
        wandb.define_metric(name='batch', step_sync=True, hidden=True, summary='none')

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if llm is not None:
                completions = unpack_vllm_output(llm.generate(batch['prompts']))
            else:
                completions = batch['answers']

            string_results, int_results, batched_matches = evaluate_batched_compute_score_single_think_trace_first_diagnosis(
                completions, batch['targets'], args.code_only
            )
            all_results = string_results | int_results

            for evaluator in evaluators:
                evaluator.update_and_log(string_results['predicted_icd_codes'], batch['target_ids'][evaluator.name], int_results)

            wandb.log(data={'batch': i}, commit=not args.log_matches_detail)
            if args.log_matches_detail:
                log_matches_and_commit(batched_matches)
            # remember all results
            all_completions.extend(completions)
            all_matches.extend(batched_matches)
            

            for metric_name, batched_metric_results in all_results.items():

                all_eval_results[metric_name].extend(batched_metric_results) if type(batched_metric_results) == list else all_eval_results[metric_name].extend([batched_metric_results])

        for evaluator in evaluators:
            evaluator.log_to_summary()

        # Saving results
        output_file = args.output_dir / ('eval_' + args.data.stem + '.parquet')

        logger.info('Saving results to %s', output_file)
        dataset_with_results = add_results_to_dataset(dataset, all_completions, all_eval_results, all_matches)
        dataset_with_results.to_parquet(output_file)

        for evaluator in evaluators:
            evaluator.save_confusion(args.output_dir / f'eval_matrix_{args.data.stem}_{evaluator.name}.json')
            #evaluator.save_non_averaged_results(args.output_dir / f'eval_NoAvg_{args.data.stem}_{evaluator.name}.json')

        logger.info('Done.')


class Evaluator:
    def __init__(self, vocab: Vocabulary,  name: str):
        self.name = name
        self.vocab = vocab
        self.num_original_labels = len(vocab)

        self.labels = vocab.labels

        self.main_metrics = create_main_diagnosis_metrics(self.num_original_labels, prefix=f'{name}/')
        self.avg_metrics = defaultdict(lambda: MeanMetric())
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=len(vocab))

    def update_and_log(self, batched_predicted_codes: list[str], target_ids: Tensor,
                       int_results: dict[str, list[int]]):

        self.update_metrics(batched_predicted_codes, target_ids)
        self.update_int_metrics(int_results)

        log_all_collections(self.main_metrics, labels=self.labels)
        wandb.log(data={name: metric.compute() for name, metric in self.avg_metrics.items()}, commit=False)

    def update_metrics(self, batched_predicted_codes: list[str], target_ids: Tensor):
        prefix = self.main_metrics.prefix

        if not prefix =='full_code/':
            batched_predicted_codes = [[pred[0][:3]] for pred in batched_predicted_codes]
            if prefix == 'chapter/':
                batched_predicted_codes = [[get_icd10_chapter(pred[0]).name] for pred in batched_predicted_codes]
        pred_indices = self.vocab.translate_batched_labels(batched_predicted_codes)
        #fake_logits = to_fake_logits(pred_indices, self.vocab)
        #pred_multi_hot = to_multi_hot(pred_indices, self.vocab)

        self.main_metrics.update(Tensor(pred_indices), target_ids)
        self.avg_metrics[prefix + '#unknown_label_predictions'].update(
            len([index  for index in pred_indices if index == (self.num_original_labels-1)]))
        #self.avg_metrics[prefix + '#predicted_icd_codes'].update([len(indices) for indices in pred_indices])

        self.confusion_matrix.update(Tensor(pred_indices), target_ids)

    def update_int_metrics(self, int_results: dict[str, list[int]]):
        for metric, values in int_results.items():
            self.avg_metrics[metric].update(values)

    def update_md_metrics(self, batched_predicted_codes: list[str], first_codes: Tensor):
        pred_indices = self.md_vocab.safe_translate_batched_labels(batched_predicted_codes)
        fake_logits = to_fake_logits(pred_indices, self.md_vocab)
        self.main_metrics.update(fake_logits, first_codes)

    def log_to_summary(self):
        for metric_name, value in self.main_metrics.items():
            if 'NoAvg' in metric_name:
                final_value = value.compute()
                buckets = final_value.tensor_split(100)
                percentile = [bucket.mean() for bucket in buckets]
                wandb.summary[metric_name.replace('NoAvg', 'Perc')] = percentile

                buckets = final_value.tensor_split(10)
                decile = [bucket.mean() for bucket in buckets]
                wandb.summary[metric_name.replace('NoAvg', 'Decile')] = decile

    def save_confusion(self, file_name):
        logger.info('Saving Confusion Matrix Table to %s', file_name)
        with open(file_name, 'w+') as f:
            json.dump(to_serializable(self.confusion_matrix.compute()), f)

    def save_non_averaged_results(self, file_name):
        logger.info('Saving Non-Averaged Results to %s', file_name)
        non_averaged_results = get_non_averaged_results(self.metrics, labels=self.labels)
        with open(file_name, 'w+') as f:
            json.dump(to_serializable(non_averaged_results), f)


def set_llm_to_min_max_tokens_or_default(llm, min_max_tokens):
    max_tokens = llm.get_default_sampling_params().max_tokens
    if max_tokens < min_max_tokens:
        logger.warning(f'Default max_tokens only at {max_tokens} tokens, setting to {min_max_tokens}')
    llm.default_sampling_params['max_tokens'] = min_max_tokens


def make_collate_fn(tokenizer, prompt_col, target_col, vocabs):
    def collate_fn(examples):
        
        prompts = [tokenizer.apply_chat_template(x[prompt_col], tokenize=False, continue_final_message=True) for x in
                   examples]
        targets = [x[target_col] for x in examples]

        return {'prompts': prompts,
                'targets': targets,
                'target_ids': {name: torch.tensor([vocab[x[target_col][0]] for x in examples])
                                for name, vocab in vocabs.items()},
                }

    return collate_fn


class BatchedDatasetLoader:
    def __init__(self, dataset, batch_size, answer_col, target_col, vocabs, md_vocabs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.answer_col = answer_col
        self.target_col = target_col
        self.vocabs = vocabs
        self.md_vocabs = md_vocabs

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]  # dict of tensors
            yield {
                'answers': batch[self.answer_col],
                'targets': batch[self.target_col],
                'target_ids': {name: to_multi_hot(vocab.translate_batched_labels(batch[self.target_col]), vocab)
                               for name, vocab in self.vocabs.items()},
                'first_codes': {name: torch.tensor([vocab[x[0]] for x in batch[self.target_col]])
                                for name, vocab in self.md_vocabs.items()},
            }

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)


def pack_to_vllm_format(texts: list[str]):
    return [SimpleNamespace(outputs=[SimpleNamespace(text=t)]) for t in texts]


def unpack_vllm_output(outputs: list) -> list[str]:
    completions = []
    for out in outputs:
        # out.outputs[0].token_ids is often a numpy array or torch tensor
        token_ids = [int(t) for t in out.outputs[0].token_ids]
        text = out.outputs[0].text
        completions.append(text)
    return completions



def merge_previous_results_to_dataset(dataset, previous_results) -> Dataset:
    # to add the prompts to the resulting dataset, for idempotency
    if 'subject_id' not in dataset.column_names:
        other_df = previous_results.select_columns(['hadm_id', 'subject_id', 'prompt']).to_pandas()
    else:
        other_df = previous_results.select_columns(['hadm_id', 'prompt']).to_pandas()
    return Dataset.from_pandas(dataset.to_pandas().merge(other_df, on='hadm_id'))


def log_matches_and_commit(batched_matches: list[list[MatchWithTraceAndQuery]]):
    """
    We need to commit every match to get a different table for each.
    """
    for matches in batched_matches:
        columns = [*MatchWithTraceAndQuery.__annotations__.keys(), 'prediction_f', 'ground_truth_f']
        data = [[*astuple(match), *_format_match(match)] for match in matches]
        wandb.log({'match': Table(columns=columns, data=data,
                                  dtype=[_dtypes.ListType(str), _dtypes.ListType(str), str, str, int, str, str])})


def add_results_to_dataset(dataset, all_completions, all_eval_results, all_matches):
    dataset = dataset.add_column('answers', all_completions)
    dataset = dataset.add_column('matches', [[asdict(match) for match in matches] for matches in all_matches])
    for metric_name, metric_results in all_eval_results.items():
        logger.debug('Adding column %s to dataset...', metric_name)
        if metric_name.startswith('#partial_matches_'):
            if len(metric_results) != len(dataset):
                metric_results = ([0] * (len(dataset) - len(metric_results))) + metric_results

        dataset = dataset.add_column(metric_name, metric_results)
    return dataset

def evaluate_batched_compute_score_single_think_trace_first_diagnosis(solution_strs: list[str], batched_ground_truths: list[Iterable[str]], code_only: bool
) -> tuple[dict[str, list[list[str]] | list[str]], dict[str, list[int]], list[list[MatchWithTraceAndQuery]]]:
    if code_only:
        regex_matches = [CODE_ONLY_PATTERN.findall(solution_str) for solution_str in solution_strs]
        first_trace_matches = []
        for match in regex_matches:
            if match != []:
                first_trace_matches.append([("", match[0].strip())])
            else:
                first_trace_matches.append([""])

    else:
        regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                        for solution_str in solution_strs]
        first_trace_matches = [[match[0]] if len(match) > 0 else [""] for match in regex_matches ]

    batched_matches = [match_model_predictions_to_ground_truth(first_trace_and_prediction, ground_truth) for
                       first_trace_and_prediction, ground_truth in zip(first_trace_matches, batched_ground_truths)]
    batched_icd_codes = [[match[0][1].strip().replace(".", "")] if match != [""] else match for match in first_trace_matches ]
    batched_traces = [[match[0][0].strip()] if match != [""] else match for match in first_trace_matches]
    batched_successful_predictions = [[match for match in matches if match.prediction is not None]
                                      for matches in batched_matches]
    batched_successful_matches = [[match for match in matches if match.ground_truth is not None]
                                  for matches in batched_successful_predictions]

    str_results = {
        'generated_output': solution_strs,
        'predicted_icd_codes': batched_icd_codes,

        'predicted_icd_codes_f':
            [[format_predictions(icd_code, successful_predictions) for icd_code in icd_codes]
             for icd_codes, successful_predictions in zip(batched_icd_codes, batched_successful_predictions)],
        'ground_truths_f':
            [[format_ground_truth(gt, successful_matches) for gt in ground_truths]
             for ground_truths, successful_matches in zip(batched_ground_truths, batched_successful_matches)],
    }

    int_results = {
                      '#traces': [len(traces) for traces in batched_traces],
                      '#trace_chars': [sum([len(trace) for trace in traces]) for traces in batched_traces],
                      '#output_chars': [len(solution_str) for solution_str in solution_strs],
                  } #| _batched_classify_and_count_model_matches(batched_matches)
    
    return str_results, int_results, batched_matches

def evaluate_batched_compute_score_multiple_think_traces(
        solution_strs: list[str], batched_ground_truths: list[Iterable[str]]
) -> tuple[dict[str, list[list[str]] | list[str]], dict[str, list[int]], list[list[MatchWithTraceAndQuery]]]:
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]

    batched_traces = [[thinking_trace for thinking_trace, _ in match_groups] for match_groups in regex_matches]
    batched_diagnoses = [[diagnosis for _, diagnosis in match_groups] for match_groups in regex_matches]
    retriever_results = batched_msearch_meilisearch(regex_matches)
    
    batched_matches = [match_predictions_to_ground_truth(icd_codes, ground_truth, traces) for
                       icd_codes, ground_truth, traces in zip(retriever_results, batched_ground_truths, batched_traces)]

    batched_icd_codes = [list(set(match['hits'][0]['icd_code'].strip() for match in results if len(match['hits']) > 0))
                         for results in retriever_results]

    batched_query_matches = [[match for match in matches if match.query is not None]
                             for matches in batched_matches]
    batched_successful_predictions = [[match for match in matches if match.prediction is not None]
                                      for matches in batched_query_matches]
    batched_successful_matches = [[match for match in matches if match.ground_truth is not None]
                                  for matches in batched_successful_predictions]

    str_results = {
        'generated_output': solution_strs,
        'predicted_icd_codes': batched_icd_codes,

        'predicted_diagnoses_f':
            [[format_diagnosis(diagnosis, query_matches) for diagnosis in diagnoses]
             for diagnoses, query_matches in zip(batched_diagnoses, batched_query_matches)],
        'predicted_icd_codes_f':
            [[format_predictions(icd_code, successful_predictions) for icd_code in icd_codes]
             for icd_codes, successful_predictions in zip(batched_icd_codes, batched_successful_predictions)],
        'ground_truths_f':
            [[format_ground_truth(gt, successful_matches) for gt in ground_truths]
             for ground_truths, successful_matches in zip(batched_ground_truths, batched_successful_matches)],
    }

    int_results = {
                      '#traces': [len(traces) for traces in batched_traces],
                      '#diagnoses': [len(diagnoses) for diagnoses in batched_diagnoses],
                      '#trace_chars': [sum([len(trace) for trace in traces]) for traces in batched_traces],
                      '#output_chars': [len(solution_str) for solution_str in solution_strs],
                  } | _batched_classify_and_count_matches(batched_matches)
    
    return str_results, int_results, batched_matches

def match_model_predictions_to_ground_truth(first_trace_and_prediction: list[Iterable[tuple[str]]],
                                      ground_truth: Iterable[str],
                                      ) -> list[MatchWithTraceAndPrediction]:
    remaining_ground_truths = {str(x).strip() for x in ground_truth}
    prediction_matches: list[MatchWithTraceAndPrediction] = []
    not_fully_matching_codes = []
    preds2traces: dict[str | None, list[str]] = defaultdict(list)
    for match in first_trace_and_prediction:
        if match == "":
            preds2traces[None].append("")
        else:
            trace = match[0].strip()
            prediction = match[1].strip().replace(".", "")
            if not bool(re.match(ICD_10_CM_PATTERN, prediction)):
                # no hit in database
                preds2traces[None].append(trace)
            else:
                
                preds2traces[prediction].append(trace)

                if prediction in remaining_ground_truths:
                    # Full code match
                    prediction_matches.append(
                        MatchWithTraceAndPrediction(preds2traces[prediction],
                                            prediction, prediction, len(prediction)))
                    remaining_ground_truths.remove(prediction)
                else:
                    not_fully_matching_codes.append(prediction)

    if None in preds2traces:
        # add queries without a match
        prediction_matches.append(MatchWithTraceAndPrediction(preds2traces[None]))

    prediction_matches += _get_partial_model_matches(not_fully_matching_codes, remaining_ground_truths, preds2traces,
                                          )

    return prediction_matches

def match_predictions_to_ground_truth(match_responses: Iterable[dict[str, str | list[dict]]],
                                      ground_truth: Iterable[str],
                                      traces: Iterable[str],
                                      ) -> list[MatchWithTraceAndQuery]:
    remaining_ground_truths = {str(x).strip() for x in ground_truth}
    query_matches: list[MatchWithTraceAndQuery] = []
    not_fully_matching_codes = []
    preds2queries: dict[str | None, list[str]] = defaultdict(list)
    preds2traces: dict[str | None, list[str]] = defaultdict(list)
    for match_response, trace in zip(match_responses, traces):
        hits = match_response['hits']
        if len(hits) == 0:
            # no hit in database
            preds2queries[None].append(match_response['query'])
            preds2traces[None].append(trace)
        else:
            predicted_icd_code = hits[0]["icd_code"].strip()
            preds2queries[predicted_icd_code].append(match_response['query'])
            preds2traces[predicted_icd_code].append(trace)

            if predicted_icd_code in remaining_ground_truths:
                # Full code match
                query_matches.append(
                    MatchWithTraceAndQuery(preds2traces[predicted_icd_code], preds2queries[predicted_icd_code],
                                           predicted_icd_code, predicted_icd_code, len(predicted_icd_code)))
                remaining_ground_truths.remove(predicted_icd_code)
            else:
                not_fully_matching_codes.append(predicted_icd_code)

    if None in preds2queries:
        # add queries without a match
        query_matches.append(MatchWithTraceAndQuery(preds2traces[None], preds2queries[None]))

    query_matches += _get_partial_matches(not_fully_matching_codes, remaining_ground_truths, preds2traces,
                                          preds2queries)

    return query_matches


def truncate_labels_10(labels: list[str]) -> list[str]:
    return [truncate_label_10(label) for label in labels]


def truncate_label_10(label: str) -> str:
    return label[:3]

def _get_partial_model_matches(predicted_codes: list[str],
                         remaining_ground_truths: set[str],
                         preds2traces: dict[str | None, list[str]],
                         ):
    prediction_matches = []
    predictions_by_first_character = separate_by_first_character_with_duplicates(predicted_codes)
    ground_truths_by_first_character = separate_by_first_character(remaining_ground_truths)

    for first_character, remaining_predictions in predictions_by_first_character.items():
        if first_character in ground_truths_by_first_character:
            sorted_matches = sort_by_highest_reward(
                [MatchWithTraceAndPrediction(preds2traces[prediction],
                                        prediction, gt, longest_common_prefix(gt, prediction))
                 for prediction in remaining_predictions for gt in ground_truths_by_first_character[first_character]]
            )

            for match in sorted_matches:
                if match.ground_truth in remaining_ground_truths and match.prediction in remaining_predictions:
                    prediction_matches.append(match)

                    remaining_predictions.remove(match.prediction)
                    remaining_ground_truths.remove(match.ground_truth)

                    if len(remaining_ground_truths) == 0 or len(remaining_predictions) == 0:
                        break

        prediction_matches += [MatchWithTraceAndPrediction(preds2traces[prediction], prediction) for
                          prediction in
                          remaining_predictions]
    return prediction_matches

def _get_partial_matches(predicted_codes: list[str],
                         remaining_ground_truths: set[str],
                         preds2traces: dict[str | None, list[str]],
                         preds2queries: dict[str | None, list[str]]):
    query_matches = []
    predictions_by_first_character = separate_by_first_character_with_duplicates(predicted_codes)
    ground_truths_by_first_character = separate_by_first_character(remaining_ground_truths)

    for first_character, remaining_predictions in predictions_by_first_character.items():
        if first_character in ground_truths_by_first_character:
            sorted_matches = sort_by_highest_reward(
                [MatchWithTraceAndQuery(preds2traces[prediction], preds2queries[prediction],
                                        prediction, gt, longest_common_prefix(gt, prediction))
                 for prediction in remaining_predictions for gt in ground_truths_by_first_character[first_character]]
            )

            for match in sorted_matches:
                if match.ground_truth in remaining_ground_truths and match.prediction in remaining_predictions:
                    query_matches.append(match)

                    remaining_predictions.remove(match.prediction)
                    remaining_ground_truths.remove(match.ground_truth)

                    if len(remaining_ground_truths) == 0 or len(remaining_predictions) == 0:
                        break

        query_matches += [MatchWithTraceAndQuery(preds2traces[prediction], preds2queries[prediction], prediction) for
                          prediction in
                          remaining_predictions]
    query_matches.append(MatchWithTraceAndQuery([], [], None, ', '.join(remaining_ground_truths)))
    return query_matches

def _batched_classify_and_count_model_matches(batched_matches: list[list[MatchWithTraceAndPrediction]]) -> dict:
    counter = {key:[value] for key, value in _classify_and_count_model_matches(batched_matches).items()}
    return counter
    #batched_counter = _classify_and_count_model_matches(batched_matches)
    
    #return {matches_type: [counter.get(matches_type, 0) for counter in batched_counter]
    #        for matches_type in batched_counter.keys()}

def _classify_and_count_model_matches(matches: list[MatchWithTraceAndPrediction]) -> dict:
    counter = {
        '#full_matches': 0,
        '#partial_matches': defaultdict(lambda: defaultdict(lambda: 0)),
        '#no_matches': 0,
        '#no_retrieval_matches': 0,
        '#unmatched_ground_truths': 0
    }
    matches = [x for xs in matches for x in xs]
    for match in matches:
        if match.prediction is None:
            
            counter['#unmatched_ground_truths'] += len(match.ground_truth.split(','))

        else:
            if match.prediction is match.ground_truth:
                counter['#full_matches'] += 1
            elif match.ground_truth is not None:
                counter['#partial_matches'][match.overlap][len(match.ground_truth)] += 1
            else:
                counter['#no_matches'] += 1

    counter |= {f'#partial_matches_{overlap}of{gt_length}': count
                for overlap, length_dict in counter['#partial_matches'].items()
                for gt_length, count in length_dict.items()}
    counter['#partial_matches'] = sum([count
                                       for length_dict in counter['#partial_matches'].values()
                                       for count in length_dict.values()])
    return counter
def _batched_classify_and_count_matches(batched_matches: list[list[MatchWithTraceAndQuery]]) -> dict:
    batched_counter = [_classify_and_count_matches(matches) for matches in batched_matches]
    return {matches_type: [counter.get(matches_type, 0) for counter in batched_counter]
            for matches_type in batched_counter[0].keys()}

def _classify_and_count_matches(matches: list[MatchWithTraceAndQuery]) -> dict:
    counter = {
        '#full_matches': 0,
        '#partial_matches': defaultdict(lambda: defaultdict(lambda: 0)),
        '#no_matches': 0,
        '#no_retrieval_matches': 0,
        '#unmatched_ground_truths': 0
    }

    for match in matches:
        if match.prediction is None:
            if len(match.query) > 0:
                counter['#no_retrieval_matches'] += len(match.query)
            else:
                counter['#unmatched_ground_truths'] += len(match.ground_truth.split(','))

        else:
            if match.prediction is match.ground_truth:
                counter['#full_matches'] += 1
            elif match.ground_truth is not None:
                counter['#partial_matches'][match.overlap][len(match.ground_truth)] += 1
            else:
                counter['#no_matches'] += 1

    counter |= {f'#partial_matches_{overlap}of{gt_length}': count
                for overlap, length_dict in counter['#partial_matches'].items()
                for gt_length, count in length_dict.items()}
    counter['#partial_matches'] = sum([count
                                       for length_dict in counter['#partial_matches'].values()
                                       for count in length_dict.values()])
    return counter


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()
    logging.info(f"Script arguments: {script_args}")
    main(script_args)
