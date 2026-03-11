from itertools import chain
from typing import Iterable, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import fixture, approx

from rewards import compute_partial_digit_overlap_reward, compute_score, verl_compute_score_multiple_think_traces, \
    verl_batched_compute_score_multiple_think_traces_and_length


class TestComputeScore:
    @fixture
    def valid_solution_str(self):
        return '''<think>I'm thinking something smart.</think>
<diagnosis>Diagnosis1</diagnosis>
    <diagnosis>Diagnosis2</diagnosis>
<diagnosis>Diagnosis3</diagnosis>'''

    @patch('rewards.RETRIEVER_CLIENT.multi_search')
    def test_compute_score_green_path(self, msearch_mock: MagicMock, valid_solution_str):
        msearch_results = ['abc', 'def', 'xyz']
        msearch_mock.return_value = _format_melli_search_results(msearch_results)

        assert compute_score(valid_solution_str, np.array(msearch_results), format_score=1000, score=10) == 1030


def _format_melli_search_results(results: Iterable[str]) -> dict:
    return {'results': [
        {'hits': [{'icd_code': r}]} if r is not None else {'hits': []} for r in results
    ]}


@fixture
def valid_solution_str():
    return '''I'm thinking something smart.</think>
<diagnosis>Diagnosis1</diagnosis>
<think>I'm thinking something smart.</think>
<diagnosis>Diagnosis2</diagnosis>
<think>I'm thinking something smart.</think>
<diagnosis>Diagnosis3</diagnosis>'''


@fixture
def ground_truth_mocks():
    msearch_results = ['abc', 'def', 'ghi']
    return _format_melli_search_results(msearch_results), np.array(msearch_results)


@fixture
def reward_scaling():
    return dict(
        length_malus=1_00_00,
        score=.01,
        trace_reward=1_00.,
        malus=1.
    )


def decode_score(actual_score, reward_scaling):
    actual_matches = (actual_score % 1) / reward_scaling['score']
    encoded_reward_str = str(round(actual_score))
    len_str = len(encoded_reward_str)

    actual_no_matches = float(encoded_reward_str[len_str - 2:len_str])
    actual_traces = float(encoded_reward_str[len_str - 4:len_str - 2])
    actual_length = float(encoded_reward_str[:len_str - 4])

    return actual_matches, actual_no_matches, actual_traces, actual_length


def mock_output_with_think_trace(diagnoses: list[str]):
    string = 'I\'m thinking something smart.</think>\n<diagnosis>%s</diagnosis>'
    return '\n'.join([string % diagnoses[0]]
                     + ['<think>' + string % diagnosis for diagnosis in diagnoses[1:]])

def _format_queries_for_meilisearch_msearch(queries: list[str]) -> Iterable[dict[str, Any]]:
    return [{'indexUid': 'icd_10_descriptions', 'q': q, 'limit': 1} for q in queries]

@patch('wandb.log')
@patch('rewards.RETRIEVER_CLIENT.multi_search')
class TestBatchedComputeScoreMultipleThinkTraces:
    def test_different_predictions(self, msearch_mock: MagicMock, wandb_mock: MagicMock, reward_scaling):
        cases = [
            (1 + 1 / 3 + 2 / 3, 0, ['axx', 'dex', 'ghi']),
            (2, 0, ['abc', 'def']),
            (3, 1, ['abc', 'def', 'ghi', 'wrong']),
            (2, 1, ['abc', 'def', 'wrong']),
            (2, 1, ['abc', 'def', 'def'])
        ]

        valid_solution_strs = [mock_output_with_think_trace(case[2]) for case in cases]
        ground_truths = [np.array(['abc', 'def', 'ghi'])] * len(cases)

        flattened_diagnoses = list(chain.from_iterable((case[2] for case in cases)))
        msearch_mock.return_value = _format_melli_search_results(flattened_diagnoses)

        actual_scores = verl_batched_compute_score_multiple_think_traces_and_length(valid_solution_strs, ground_truths,
                                                                                    **reward_scaling)

        assert msearch_mock.call_count == 1
        msearch_mock.assert_called_with(_format_queries_for_meilisearch_msearch(flattened_diagnoses))

        for actual_score, (overlap, wrong, predictions), valid_solution_str, ground_truth in \
                zip(actual_scores, cases, valid_solution_strs, ground_truths):
            actual_matches, actual_no_matches, actual_traces, actual_length = decode_score(actual_score, reward_scaling)
            assert actual_matches == approx(overlap), f'in: {predictions} -> gt: {ground_truth.tolist()}'
            assert actual_no_matches == wrong, f'in: {predictions} -> gt: {ground_truth.tolist()}'
            assert actual_traces == len(predictions), f'in: {predictions} -> gt: {ground_truth.tolist()}'
            assert actual_length == len(valid_solution_str), f'in: {predictions} -> gt: {ground_truth.tolist()}'


@patch('wandb.log')
@patch('rewards.RETRIEVER_CLIENT.multi_search')
class TestComputeScoreMultipleThinkTraces:
    def test_no_output(self, msearch_mock: MagicMock, wandb_mock: MagicMock, ground_truth_mocks: MagicMock):
        assert verl_compute_score_multiple_think_traces('Some stupid output', ground_truth_mocks[1]) == 0.0

    def test_green_path(self, msearch_mock: MagicMock, wandb_mock: MagicMock, ground_truth_mocks: MagicMock):
        msearch_formatted, ground_truth = ground_truth_mocks
        msearch_mock.return_value = msearch_formatted
        valid_solution_str = mock_output_with_think_trace(ground_truth.tolist())

        assert (verl_compute_score_multiple_think_traces(valid_solution_str, ground_truth, score=10, trace_reward=1000,
                                                         length_malus=0.1)
                == 3030.0 + len(valid_solution_str) * 0.1)

        assert msearch_mock.call_count == 1
        msearch_mock.assert_called_with(_format_queries_for_meilisearch_msearch(ground_truth.tolist()))

    @pytest.mark.parametrize('traces,completion', [
        (2, '''<think>I'm thinking something smart.</think>
    <diagnosis>Diagnosis1</diagnosis>
    <think>I'm thinking something smart.</think>
        <diagnosis>Diagnosis2</diagnosis>
    <think>I'm thinking something smart.</think>
    <diagnosis>Diagnosis3</diagnosis>'''),
        (2, '''I'm thinking something smart.</think>
<diagnosis>Diagnosis1</diagnosis>
    <diagnosis>Diagnosis2</diagnosis>
<think>I'm thinking something smart.</think>
<diagnosis>Diagnosis3</diagnosis>'''),
        (0, '''
<diagnosis>Diagnosis1</diagnosis>
    <diagnosis>Diagnosis2</diagnosis>
<diagnosis>Diagnosis3</diagnosis>''')], ids=['leading_think_tag', '1_missing', 'all_missing'])
    def test_incorrect_traces(self, msearch_mock: MagicMock, wandb, reward_scaling, ground_truth_mocks: MagicMock,
                              traces, completion):
        msearch_formatted, ground_truth = ground_truth_mocks
        msearch_mock.return_value = msearch_formatted
        actual_score = verl_compute_score_multiple_think_traces(completion, ground_truth, **reward_scaling)
        actual_matches, actual_no_matches, actual_traces, actual_length = decode_score(actual_score, reward_scaling)

        assert actual_matches == approx(3)
        assert actual_no_matches == 0
        assert actual_traces == traces
        assert actual_length == len(completion)

        # sanity check
        assert actual_score == len(completion) * reward_scaling['length_malus'] + 3 * reward_scaling[
            'score'] + traces * reward_scaling['trace_reward'] + 0 * reward_scaling['malus']

    @pytest.mark.parametrize('overlap, wrong, search_results', [
        (1 + 1 / 3 + 2 / 3, 0, ['axx', 'dex', 'ghi']),
        (2, 0, ['abc', 'def']),
        (3, 1, ['abc', 'def', 'ghi', 'wrong']),
        (2, 1, ['abc', 'def', 'wrong']),
        (2, 1, ['abc', 'def', 'def'])
    ], ids=['increasingly_more_overlap', 'one_missing_prediction', 'one_too_many_predictions',
            'one_wrong_prediction', 'duplicated_prediction']
                             )
    def test_different_predictions(self, msearch_mock: MagicMock, wandb_mock: MagicMock, valid_solution_str,
                                   overlap, search_results: list[str], wrong: int, reward_scaling):
        ground_truth = np.array(['abc', 'def', 'ghi'])
        msearch_mock.return_value = _format_melli_search_results(search_results)
        actual_score = verl_compute_score_multiple_think_traces(valid_solution_str, ground_truth, **reward_scaling)
        actual_matches, actual_no_matches, actual_traces, actual_length = decode_score(actual_score, reward_scaling)

        assert actual_matches == approx(overlap)
        assert actual_no_matches == wrong
        assert actual_traces == 3
        assert actual_length == len(valid_solution_str)

    def test_no_hit(self, msearch_mock: MagicMock, wandb_mock: MagicMock, valid_solution_str, reward_scaling):
        ground_truth = np.array(['abc', 'def', 'ghi'])
        msearch_mock.return_value = _format_melli_search_results(['abc', None, None])

        actual_score = verl_compute_score_multiple_think_traces(valid_solution_str, ground_truth, **reward_scaling)
        actual_matches, actual_no_matches, actual_traces, actual_length = decode_score(actual_score, reward_scaling)

        assert actual_matches == approx(1)
        assert actual_no_matches == 2
        assert actual_traces == 3
        assert actual_length == len(valid_solution_str)

    def test_for_reward_hacking(self, msearch_mock: MagicMock, wandb):
        solution_str = ("I'm thinking something smart.</think>\n<diagnosis>Diagnosis1</diagnosis>\n"
                        + "<think>I'm thinking something smart.</think>\n<diagnosis>Diagnosis2</diagnosis>")
        ground_truth = np.array(['abc', 'def', 'ghi'])
        msearch_mock.return_value = _format_melli_search_results(['abc'])
        valid_reward = verl_compute_score_multiple_think_traces(solution_str, ground_truth, length_malus=0.)

        hacked_solution_str = ("I'm thinking something smart.</think>\n<diagnosis>Diagnosis1</diagnosis>\n"
                               + "<think>I'm thinking something smart.</think><diagnosis>Diagnosis2</diagnosis>" * 60)
        msearch_mock.return_value = _format_melli_search_results(['abc'] + [None] * 60)
        hacked_reward = verl_compute_score_multiple_think_traces(hacked_solution_str, ground_truth, length_malus=0.)

        assert valid_reward > hacked_reward


class TestComputePartialOverlapReward:

    @pytest.mark.parametrize('predictions, ground_truth, overlap, no_matches', [
        (['abc', '1234', '12'], ['abcde', '1234'], 3 / 5 + 4 / 4, 1),
        (['abcde', '1234'], ['abc', '1234', '12'], 3 / 3 + 4 / 4, 0),
        (['abc', '123'], ['abc', '123'], 1 + 1, 0),
        (['abc', 'efg'], ['abcd', 'efghi'], 3 / 4 + 3 / 5, 0),
        (['xyz'], ['abc'], 0, 1),
        ([], ['abc'], 0, 0),
        (['abc'], [], 0, 1),
        (['abcd', 'abcd', 'abcd'], ['abcd', 'abce'], 4 / 4 + 3 / 4, 1),
    ], ids=['basic case #1', 'basic case #2', 'exact_matches', 'partial_prefixes', 'non_matching_strings',
            'empty_preds', 'empty_gt', 'duplicated_predictions']
                             )
    def test_cases(self, predictions: list[str], ground_truth: list[str], overlap, no_matches):
        actual_overlap, actual_no_matches = compute_partial_digit_overlap_reward(predictions, ground_truth)
        assert actual_overlap == overlap
        assert actual_no_matches == no_matches

    @fixture
    def sample_data(self):
        return ['a', 'abcde', '12345', '1236', '1234', '123', 'xyz'], ['abcdf', 'abc', '12345', '12388']

    def test_hard_case_forward(self, sample_data):
        predicted, ground_truth = sample_data
        overlap, no_matches = compute_partial_digit_overlap_reward(predicted, ground_truth)
        assert overlap == approx(1 / 3 + 4 / 5 + 5 / 5 + 3 / 5)
        assert no_matches == 3

    def test_hard_case_backward(self, sample_data):
        ground_truth, predicted = sample_data
        overlap, no_matches = compute_partial_digit_overlap_reward(predicted, ground_truth)
        assert overlap == approx(4 / 5 + 1 / 1 + 5 / 5 + 3 / 3)
        assert no_matches == 0

    def test_is_order_invariant(self):
        predictions = ['a', 'abc', 'abcdef']
        ground_truth = ['abc', 'abcdef']

        overlap, no_matches = compute_partial_digit_overlap_reward(predictions, ground_truth)
        assert overlap == 2
        assert no_matches == 1

        overlap, no_matches = compute_partial_digit_overlap_reward(reversed(predictions), ground_truth)
        assert overlap == 2
        assert no_matches == 1

        overlap, no_matches = compute_partial_digit_overlap_reward(predictions, reversed(ground_truth))
        assert overlap == 2
        assert no_matches == 1

        overlap, no_matches = compute_partial_digit_overlap_reward(reversed(predictions), reversed(ground_truth))
        assert overlap == 2
        assert no_matches == 1
