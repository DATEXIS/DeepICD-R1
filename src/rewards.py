import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from itertools import islice
from re import Pattern
from typing import Any, AnyStr, Iterable
from urllib.error import HTTPError

import httpx
import meilisearch
import numpy as np
import requests
from aiohttp import ClientSession, ClientError
from openai import OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

# VllmClient is inlined here because verl requires the reward module to be
# self-contained in a single file (no relative imports at training time).
class LLMScore(BaseModel):
    score: int
class VllmClient:
    def __init__(self,
                 api_base,
                 api_key,
                 batch_size = 8,
                 concurrency = 16):
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=httpx.Timeout(1000000),
        )

    def get_api_config(self) -> dict:
        # on own namespace: http://vllm-server:80/v1
        return {
            "openai_api_key": self.api_key,
            'openai_api_base': self.api_base,
            'openai_api_health_url': self.api_base[:-2] + "health",
        }


    def check_connection(self, api_config):
        backoff_time = 1  # Start with 1 second
        num_tries = 0
        max_tries = 100
        while num_tries <= max_tries:
            try:
                response = requests.get(api_config['openai_api_health_url'])
                if response.status_code == 200:
                    return True
            except (requests.exceptions.RequestException, HTTPError):
                logging.info(f"Connection attempt {num_tries}, retrying in {backoff_time}s...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)  # Exponential backoff (capped at 60s)
                num_tries += 1
        raise RuntimeError(f"Could not connect to vLLM server after {max_tries} attempts")


    def get_request(self, session, scoring_prompts, model, api_config):
        url = f"{api_config['openai_api_base']}/completions"
        payload = {
            "prompt": scoring_prompts,
            "temperature": 0.2,
            "guided_choice": ["1","2","3","4","5"],
            "model": model,
            "echo": False,
            "stream": False,
            "guided_decoding_backend": "guidance"
        }
        headers = {"Content-Type": "application/json"}
        return session.post(url, json=payload, headers=headers)


    async def gather_with_concurrency(self, n, *coros):
        semaphore = asyncio.Semaphore(n)

        async def sem_coro(coro):
            async with semaphore:
                try:
                    response = await coro
                    if hasattr(response, "status") and response.status >= 400:
                        logging.error(f"HTTP error {response.status}: {await response.text()}")
                        return None  # Handle bad r
                    return response
                except (ClientError, asyncio.TimeoutError, ConnectionResetError) as e:
                    logging.error(f"Request failed: {e}")
                    return None  # Return None instead of crashing
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    return None

        return await tqdm.gather(*(sem_coro(c) for c in coros))


    def get_model(self) -> str:
        
        models = self.client.models.list()
        logging.info(f"Got model ID: {models.data[0].id}")
        return models.data[0].id



    def send_prompts(self, prompts: list, session: ClientSession,
                        llm_name: str, api_config: dict) -> list:
        # Convert prompts to prompt batches
        logging.info(f"Convert prompts to batches of size {self.batch_size}")
        batches = [prompts[i:i + self.batch_size]
                for i in range(0, len(prompts), self.batch_size)]

        # Convert prompt batches to request batches
        responses = [self.get_request(session, batch, llm_name, api_config) for batch in batches]

        # Send batches with concurrency and wait until all responses arrived
        return responses


    def extract_text_from_responses(self, responses: list, num_choices: int) -> list:
        # Extract text from responses
        responses = [(response.json()) for response in responses]
        responses = [response["choices"] for response in responses]

        if num_choices == 1:
            # flatten choices as there is only one option, return [choice1, choice1, choice1, ...]
            return [response[0]['text'] for response in responses]
        else:
            # Remove batch structure
            responses = [response for batch in responses for response in batch]
            responses = [responses[i:i + num_choices] for i in range(0, len(responses), num_choices)]
            # Return [[choice1, choice2, ...], [choice1, choice2, ...]]
            return [[choice['text'] for choice in response] for response in responses]


    def score_traces(self,prompts: list) -> list:

        api_config = self.get_api_config()
        self.check_connection(api_config)
    
        llm_name = self.get_model()
       
        response = self.client.completions.create(prompt=prompts,
                                                    model=llm_name,
                                                    stream=False,
                                                    #extra_body={
                                                    #"guided_choice":["1","2","3","4","5"],
                                                    #"guided_decoding_backend": "guidance"}
                                                    )
                                                    
        scores = []
        for choice in response.choices:
            try:
                m = re.search(r'\{[\s\S]*?\}', choice.text)
                scores.append(json.loads(m[0])["score"])
            except ValueError as e:
                print("JSON error")
                scores.append(0)
        return scores

"""

ACTUAL REWARD FUNCTIONS

"""
JUDGE_CLIENT = VllmClient(api_base=os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1"),
                          api_key=os.getenv("OPENAI_API_KEY", 'openai-abc-key'))
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

@lru_cache(maxsize=None)
def judge_tokenizer():  # lazy Singleton, to allow tests without running this
    return AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)

@lru_cache(maxsize=None)
def judge_prompts() -> dict:
    """Lazily loaded so tests can import this module without the prompt file present."""
    with open(os.getenv("JUDGE_PROMPT_FILE", "/workdir/prompts/llm_judge_rag_prompt.json"), "r") as fp:
        return json.load(fp)


@lru_cache(maxsize=None)
def icd10_cm_guidelines() -> dict:
    """Lazily loaded so tests can import this module without the guidelines file present."""
    with open(os.getenv("GUIDELINES_FILE", "/pvc/data/icd10_cm_guidelines_chapters_with_null_roman_ids.json"), "r") as fp:
        return json.load(fp)


JUDGE_RAG_ENABLED = bool(os.getenv("JUDGE_RAG_ENABLED", True))

RETRIEVER_CLIENT = meilisearch.Client(
    os.getenv('MEILISEARCH_URL', "http://localhost:7700"))

MEILI_INDEX = os.getenv("MEILI_ICD_INDEX", 'mimic_icd_10_first_diagnosis_codes')

NO_MATCH_MALUS = float(os.getenv("NO_MATCH_MALUS", -.2))
''' penalty for superfluous predictions, if the trace_reward is higher than this malus reward_hacking might happen '''

MATCH_REWARD = float(os.getenv("MATCH_REWARD", 15.))
''' reward per full match, divided by length of target code for partial reward '''

THREE_DIGIT_BONUS = float(os.getenv("3D_BONUS", 1.))

TRACE_REWARD = float(os.getenv("THINK_TRACE_REWARD", .1))
''' reward per trace '''

LENGTH_MALUS = float(os.getenv("LENGTH_MALUS", -1e-4))
''' malus per character in the output '''

NO_GEN_MALUS = float(os.getenv("NO_GEN_MALUS", -5.))
'''malus for no generation at all '''

FORMAT_REWARD = float(os.getenv("FORMAT_REWARD", 1.))
''' reward if it aligns with the format, only required for single think trace case '''

LLM_REWARD_SCALING = float(os.getenv("LLM_REWARD_SCALING", .05))

ACTIVATE_OUTCOME_REWARD = bool(os.getenv("ACTIVATE_OUTCOME_REWARD", True))

ACTIVATE_FORMAT_REWARD = bool(os.getenv("ACTIVATE_FORMAT_REWARD", True))

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

def get_icd10_chapter(code: str) -> ICD10Chapter | None:
    code = code.upper()

    for chapter in ICD10Chapter:
        if code < chapter.end:
            return chapter
    return None
#########################################
###                                   ###
### ---------- Shared code ---------- ###
###                                   ###
#########################################

# everything needs to be in one file due to how verl loads the reward function
# so making this look clean is hard

ANY_OR_NO_WHITESPACE = r'\s*'
BEGIN_OF_STRING = r'^'


def xml_tag_without_inner_tag(tag_name: str) -> str:
    content_without_inner_tag = grouped(rf'[^<]*(?:<(?!/?{tag_name}>)[^<]*)*')
    return rf'<{tag_name}>{content_without_inner_tag}</{tag_name}>'


def grouped(string: str) -> str:
    return rf'({string})'


def optionally(string: str) -> str:
    return rf'(?:{string})?'


def at_least_once(string: str) -> str:
    return rf'(?:{string})+'


def batched_msearch_meilisearch(regex_matches: list[list[tuple[str, str]]]):
    flattened_diagnoses = [diagnosis for matched_groups in regex_matches for _, diagnosis in matched_groups]
    flattened_hits = msearch_meilisearch(flattened_diagnoses)
    iterator = iter(flattened_hits)
    batched_hits = [list(islice(iterator, len(matched_group))) for matched_group in regex_matches]
    return batched_hits

def batched_msearch_meilisearch_with_traces(regex_matches: list[list[tuple[str, str]]]):
    flattened_diagnoses = [diagnosis for matched_groups in regex_matches for _, diagnosis in matched_groups]
    flattened_traces = [trace for matched_groups in regex_matches for trace, _ in matched_groups]
    flattened_hits = msearch_meilisearch(flattened_diagnoses)

    for hit, trace in zip(flattened_hits, flattened_traces):
        hit["trace"] = trace

    iterator = iter(flattened_hits)
    batched_hits = [list(islice(iterator, len(matched_group))) for matched_group in regex_matches]
    return batched_hits



def msearch_meilisearch(queries: list[str]) -> list[dict[str, Any]]:
    queries = [{'indexUid': MEILI_INDEX, 'q': q, 'limit': 1} for q in queries]
    return RETRIEVER_CLIENT.multi_search(queries)['results']


@dataclass
class Match:
    prediction: str
    ground_truth: str
    overlap: int


@dataclass
class MatchWithTraceAndQuery:
    trace: list[str]
    query: list[str]
    prediction: str = None
    ground_truth: str = None
    overlap: int = None

    @cached_property
    def title(self):
        return f'{self.query} -> {self.prediction} -> {self.ground_truth}'


def compute_partial_digit_overlap_reward(predicted_codes: Iterable[str], ground_truths: Iterable[str]) -> tuple[
    float, int]:
    """
    Computes the reward for overlap between predicted codes and a ground truth, adding a malus for every prediction that does not get matched.
    It preserves duplicates of the predictions, as they are still valid for a partial match, while we do not assume ground truth to contain duplicates.
    """
    
    reward = 0.
    no_matches = 0

    predictions_by_first_character = separate_by_first_character_with_duplicates(predicted_codes)
    ground_truths_by_first_character = separate_by_first_character(ground_truths)

    for first_character, remaining_predictions in predictions_by_first_character.items():
        if first_character in ground_truths_by_first_character:
            remaining_ground_truths = ground_truths_by_first_character[first_character]
            sorted_matches = sort_by_highest_reward(
                (Match(pred, gt, longest_common_prefix(gt, pred))
                 for pred in remaining_predictions for gt in remaining_ground_truths)
            )

            for match in sorted_matches:
                ground_truth_code = match.ground_truth
                predicted_code = match.prediction
                if ground_truth_code in remaining_ground_truths and predicted_code in remaining_predictions:
                    reward += match.overlap / len(ground_truth_code)
                    if match.overlap >= 3:
                        reward += (match.overlap - 2)
                    remaining_ground_truths.remove(ground_truth_code)
                    remaining_predictions.remove(predicted_code)

                    if len(remaining_ground_truths) == 0 or len(remaining_predictions) == 0:
                        break

        
        no_matches += len(remaining_predictions)

    return reward, no_matches


def separate_by_first_character_with_duplicates(items: Iterable[str]):
    container = defaultdict(list)
    for item in items:
        container[item[0]].append(item)
    return container


def separate_by_first_character(items: Iterable[str]):
    container = defaultdict(set)
    for item in items:
        container[item[0]].add(item)
    return container


def sort_by_highest_reward(matches: Iterable[MatchWithTraceAndQuery | Match]) -> Iterable[
    MatchWithTraceAndQuery | Match]:
    return sorted(matches, key=lambda match: (match.overlap, -len(match.ground_truth)), reverse=True)


def longest_common_prefix(s1: str, s2: str):
    """Finds the length of the longest common prefix between two strings."""
    min_length = min(len(s1), len(s2))
    for i in range(min_length):
        if s1[i] != s2[i]:
            return i
    return min_length


###################################################
###                                             ###
### ---------- Multiple Think Traces ---------- ###
###                                             ###
###################################################


def think_trace_with_diagnosis() -> Pattern[AnyStr]:
    return re.compile(
        BEGIN_OF_STRING
        + ANY_OR_NO_WHITESPACE
        + optionally(xml_tag_without_inner_tag('think'))
        + ANY_OR_NO_WHITESPACE
        + xml_tag_without_inner_tag('diagnosis'),
        re.MULTILINE
    )


OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX = think_trace_with_diagnosis()
ICD_10_CM_PATTERN = r"^[A-Z][0-9]{2}[A-Z0-9]{0,4}$"

def trl_batched_length_malus_score(completions: list[list[dict[str, str]]], **kwargs):
    return verl_batched_length_malus_score([completion[0]['content'] for completion in completions], **kwargs)


def verl_batched_length_malus_score(solution_strs: list[str],
                                    length_malus=LENGTH_MALUS,
                                    **kwargs):
    length_scores = [len(solution_str) * length_malus for solution_str in solution_strs]
    length_scores = [score if score != 0 else NO_GEN_MALUS for score in length_scores]
    return length_scores


def trl_batched_compute_score_multiple_think_traces(completions: list[list[dict[str, str]]], reward_model, **kwargs):
    # verl format
    solution_strs = [completion[0]['content'] for completion in completions]
    ground_truths = [r['ground_truth'] for r in reward_model]

    scores = verl_batched_compute_score_multiple_think_traces(solution_strs, ground_truths, **kwargs)
    return scores

def trl_batched_traces_llm_score(completions: list[list[dict[str, str]]], reward_model,
                                **kwargs):
    solution_strs = [completion[0]['content'] for completion in completions]
    ground_truths = [r['ground_truth'] for r in reward_model]
    
    return verl_batched_compute_score_multiple_think_traces_and_length_and_llm(solution_strs, ground_truths, **kwargs)

"""

LLM SCORE W/O LENGTH MALUS

"""

def verl_batched_compute_score_multiple_think_traces_and_llm(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                length_malus=LENGTH_MALUS,
                                                                **kwargs):

    match_and_format_scores = verl_batched_compute_score_multiple_think_traces(solution_strs, ground_truths, score,
                                                                               malus,
                                                                               trace_reward)
    notes = [info["note"] for info in kwargs["extra_infos"]]
    
    llm_scores = _compute_llm_as_a_judge_score(solution_strs, notes, LLM_REWARD_SCALING)
    return [match_and_format_score  + llm_score
            for match_and_format_score, llm_score in zip(match_and_format_scores, llm_scores)]

def verl_batched_compute_score_multiple_think_traces_and_length_and_llm(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                length_malus=LENGTH_MALUS,
                                                                llm_malus=LLM_REWARD_SCALING,
                                                                **kwargs):
    
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]

    detected_icd_codes_with_traces = batched_msearch_meilisearch_with_traces(regex_matches)
    
    no_gen_rewards = [NO_GEN_MALUS if len(solution_str) == 0 else 0 for solution_str in solution_strs]

    if ACTIVATE_OUTCOME_REWARD:
        diagnosis_match_rewards = [_compute_diagnoses_match_reward(icd_codes, ground_truth, score, malus) for icd_codes, ground_truth in zip(detected_icd_codes_with_traces, ground_truths)]
    else:
        diagnosis_match_rewards = [0 for _ in solution_strs]
    
    if ACTIVATE_FORMAT_REWARD:
        format_rewards = [_compute_trace_score(matched_groups, trace_reward) for matched_groups in regex_matches]
    else:
        format_rewards = [0 for _ in solution_strs]

    match_and_format_scores = [diagnosis_match_reward 
            + format_reward
            + no_gen_reward
            for diagnosis_match_reward, format_reward, no_gen_reward in zip(diagnosis_match_rewards, format_rewards, no_gen_rewards)]
    
    length_scores = verl_batched_length_malus_score(solution_strs, length_malus)

    notes = [info["note"] for info in kwargs["extra_infos"]]
    
    llm_scores = _compute_llm_as_a_judge_score(notes, detected_icd_codes_with_traces, llm_malus)
    return [match_and_format_score + length_score + llm_score
            for match_and_format_score, length_score, llm_score in zip(match_and_format_scores, length_scores, llm_scores)]

def verl_batched_compute_score_multiple_think_traces_and_length(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                length_malus=LENGTH_MALUS,
                                                                **kwargs):
    match_and_format_scores = verl_batched_compute_score_multiple_think_traces(solution_strs, ground_truths, score,
                                                                               malus,
                                                                               trace_reward)
    length_scores = verl_batched_length_malus_score(solution_strs, length_malus)
    

    return [match_and_format_score + length_score
            for match_and_format_score, length_score in zip(match_and_format_scores, length_scores)]


def verl_batched_compute_score_multiple_think_traces(solution_strs: list[str],
                                                     ground_truths: list[Iterable[str]],
                                                     score=MATCH_REWARD,
                                                     malus=NO_MATCH_MALUS,
                                                     trace_reward=TRACE_REWARD,
                                                     **kwargs):
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]

    detected_icd_codes = batched_msearch_meilisearch(regex_matches)

    no_gen_rewards = [NO_GEN_MALUS if len(solution_str) == 0 else 0 for solution_str in solution_strs]

    if ACTIVATE_OUTCOME_REWARD:
        diagnosis_match_rewards = [_compute_diagnoses_match_reward(icd_codes, ground_truth, score, malus) for icd_codes, ground_truth in zip(detected_icd_codes, ground_truths)]
    else:
        diagnosis_match_rewards = [0 for _ in solution_strs]
    
    if ACTIVATE_FORMAT_REWARD:
        format_rewards = [_compute_trace_score(matched_groups, trace_reward) for matched_groups in regex_matches]
    else:
        format_rewards = [0 for _ in solution_strs]
    return [diagnosis_match_reward 
            + format_reward
            + no_gen_reward
            for diagnosis_match_reward, format_reward, no_gen_reward in zip(diagnosis_match_rewards, format_rewards, no_gen_rewards)]


def verl_compute_score_multiple_think_traces(solution_str: str,
                                             ground_truth: Iterable[str],
                                             score=MATCH_REWARD,
                                             malus=NO_MATCH_MALUS,
                                             trace_reward=TRACE_REWARD,
                                             length_malus=LENGTH_MALUS,
                                             **kwargs):
    matched_groups = OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
    if len(matched_groups) == 0:  # nothing found at all
        return 0.0

    # match with database
    diagnoses = [diag for thinking_trace, diag in matched_groups]
    matches = msearch_meilisearch(diagnoses)

    return (_compute_diagnoses_match_reward(matches, ground_truth, score, malus)
            + _compute_trace_score(matched_groups, trace_reward)
            + len(solution_str) * length_malus)  # malus for length


def verl_batched_compute_score_single_think_trace_and_llm_wo_meili(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                rag_enabled=JUDGE_RAG_ENABLED,
                                                                length_malus=LENGTH_MALUS,
                                                                **kwargs):
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]
    first_trace_matches = [[match[0]] if len(match) > 0 else "" for match in regex_matches ]
    match_and_format_rewards = verl_batched_compute_score_single_think_trace_wo_meili(solution_strs,
                                                                                     ground_truths,
                                                                                     score,
                                                                                     malus,
                                                                                     trace_reward,
                                                                                     kwargs)
    notes = [info["note"] for info in kwargs["extra_infos"]]
    
    llm_rewards = _compute_llm_as_a_judge_score_single_traces(notes, first_trace_matches, rag_enabled, LLM_REWARD_SCALING)
   
    return [match_and_format_reward
            + llm_reward
            for match_and_format_reward, llm_reward in zip(match_and_format_rewards, llm_rewards)]
def verl_batched_compute_score_single_think_trace_wo_meili(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                length_malus=LENGTH_MALUS,
                                                                **kwargs):
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]
    first_trace_matches = [[match[0]] if len(match) > 0 else "" for match in regex_matches ]
    exceeding_traces_rewards = [len(match[1:])*(-1.0) if len(match) > 1 else 0 for match in regex_matches]
    

    
    no_gen_rewards = [NO_GEN_MALUS if len(solution_str) == 0 else 0 for solution_str in solution_strs]

    if ACTIVATE_OUTCOME_REWARD:
        diagnosis_match_rewards = []
        for match, ground_truth in zip(first_trace_matches, ground_truths):
            if match != "":
                generated_code = match[0][1].strip().replace(".", "")
                if bool(re.match(ICD_10_CM_PATTERN, generated_code)):
                    diagnosis_match_rewards.append(_compute_model_diagnoses_reward(generated_code, ground_truth, score, malus))
                else:
                    diagnosis_match_rewards.append(-1)
            else:
                diagnosis_match_rewards.append(-1)
    else:
        diagnosis_match_rewards = [0 for _ in solution_strs]
    
    if ACTIVATE_FORMAT_REWARD:
        format_rewards = [_compute_trace_score(matched_groups, trace_reward) for matched_groups in first_trace_matches]
    else:
        format_rewards = [0 for _ in solution_strs]
    return [diagnosis_match_reward 
            + format_reward
            + no_gen_reward
            + exceeding_traces_reward
    
            for diagnosis_match_reward, format_reward, no_gen_reward, exceeding_traces_reward in zip(diagnosis_match_rewards, format_rewards, no_gen_rewards, exceeding_traces_rewards)]
    
    

def verl_batched_compute_score_single_think_trace_and_length(solution_strs: list[str],
                                                                ground_truths: list[Iterable[str]],
                                                                score=MATCH_REWARD,
                                                                malus=NO_MATCH_MALUS,
                                                                trace_reward=TRACE_REWARD,
                                                                length_malus=LENGTH_MALUS,
                                                                **kwargs):
    match_and_format_scores = verl_batched_compute_score_single_think_trace(solution_strs, ground_truths, score,
                                                                               malus,
                                                                               trace_reward)
    length_scores = verl_batched_length_malus_score(solution_strs, length_malus)
    

    return [match_and_format_score + length_score
            for match_and_format_score, length_score in zip(match_and_format_scores, length_scores)]

def verl_batched_compute_score_single_think_trace(solution_strs: list[str],
                                                     ground_truths: list[str],
                                                     score=MATCH_REWARD,
                                                     malus=NO_MATCH_MALUS,
                                                     trace_reward=TRACE_REWARD,
                                                     **kwargs):
    regex_matches = [OPTIONAL_THINK_TRACE_WITH_DIAGNOSIS_REGEX.findall("<think>" + solution_str)
                     for solution_str in solution_strs]
    first_trace_matches = [[match[0]] if len(match) > 0 else "" for match in regex_matches ]
    exceeding_traces_rewards = [len(match[1:])*(-5.0) if len(match) > 1 else 0 for match in regex_matches]
    detected_icd_codes = batched_msearch_meilisearch(first_trace_matches)

    no_gen_rewards = [NO_GEN_MALUS if len(solution_str) == 0 else 0 for solution_str in solution_strs]

    if ACTIVATE_OUTCOME_REWARD:
        diagnosis_match_rewards = [_compute_diagnoses_match_reward(icd_code, ground_truth, score, malus) for icd_code, ground_truth in zip(detected_icd_codes, ground_truths)]
    else:
        diagnosis_match_rewards = [0 for _ in solution_strs]
    
    if ACTIVATE_FORMAT_REWARD:
        format_rewards = [_compute_trace_score(matched_groups, trace_reward) for matched_groups in first_trace_matches]
    else:
        format_rewards = [0 for _ in solution_strs]
    return [diagnosis_match_reward 
            + format_reward
            + no_gen_reward
            + exceeding_traces_reward
            for diagnosis_match_reward, format_reward, no_gen_reward, exceeding_traces_reward in zip(diagnosis_match_rewards, format_rewards, no_gen_rewards, exceeding_traces_rewards)]
def _compute_llm_as_a_judge_score_single_traces(adm_notes, detected_icd_codes_with_traces, rag_enabled, malus):
    
    scores = []
    if rag_enabled:
        results = batched_msearch_meilisearch(detected_icd_codes_with_traces)
        symptoms = []
        for x in results:
            if len(x) > 0:
                if len(x[0]["hits"]) > 0:
                    if "symptoms" in x[0]['hits'][0].keys():
                        symptoms.append(x[0]['hits'][0]["symptoms"])
                    else:
                        symptoms.append("No symptoms available")
                else:
                    symptoms.append("No symptoms available")
            else:
                symptoms.append("No symptoms available")
    else: 
        symptoms = ["" for _ in range(len(detected_icd_codes_with_traces))]
    for i, (code_and_trace, note, symptom) in enumerate(zip(detected_icd_codes_with_traces, adm_notes, symptoms)):
        if len(code_and_trace) > 0:
            trace = code_and_trace[0][0].strip()
            code = code_and_trace[0][1].strip()
            if bool(re.match(ICD_10_CM_PATTERN, code)):

                chapter = icd10_cm_guidelines()[get_icd10_chapter(code[:3]).name]
                
                if len(trace) != 0:              
                    if rag_enabled: 
                        prompts = prepare_guidelines_prompts([trace], [code], note, [symptom], [chapter])
                    else:
                        prompts = prepare_prompts([trace], [code],note)

                    scores.append(JUDGE_CLIENT.score_traces(prompts)[0])
                else:
                    scores.append(-1)
            else:
                scores.append(-1)
        else:
            scores.append(-1)
    return [score*malus for score in scores]
def _compute_llm_as_a_judge_score(adm_notes, detected_icd_codes_with_traces, malus):
    
    scores = []

    for i, (codes_and_traces, note) in enumerate(zip(detected_icd_codes_with_traces, adm_notes)):
        traces = []
        codes = []
        chapters = []
        symptoms = []
        no_db_hits = 0
        for x in codes_and_traces:
            if len(x["hits"]) > 0:
                traces.append(x["trace"])
                codes.append(f"{x['hits'][0]['icd_code']} - {x['hits'][0]['description']}")
                if JUDGE_RAG_ENABLED:
                    chapters.append(icd10_cm_guidelines()[get_icd10_chapter(x['hits'][0]['icd_code']).name])
                    if "symptoms" in x['hits'][0].keys():
                        symptoms.append(x['hits'][0]["symptoms"])
                    else:
                        symptoms.append("No symptoms available")
        if len(traces) != 0:              
            if JUDGE_RAG_ENABLED:          
                prompts = prepare_guidelines_prompts(traces, codes, note, symptoms=symptoms, chapters=chapters)
            else:
                prompts = prepare_prompts(traces, codes,note)

            traces_score = JUDGE_CLIENT.score_traces(prompts)
            scores.append(sum(traces_score)*malus)
        else:
            scores.append(0)
    
    return scores
        


def prepare_guidelines_prompts(traces, codes, note, symptoms, chapters):
    messages = [[
                    {"role":"system", "content":judge_prompts()["SYSTEM"]},
                    {"role":"user", "content":judge_prompts()["USER"].replace(
                        "$$ADMISSION_NOTE$$", note
                        ).replace(
                            "$$TRACE$$", trace
                            ).replace(
                                "$$ICD$$", code
                                ).replace(
                                    "$$SYMPTOMS$$", symptom
                                    ).replace(
                                        "$$ICD_CHAPTER$$", chapter
                                        )
                                }
            ] for code, trace, symptom, chapter in zip(codes, traces, symptoms, chapters)]
    
    return judge_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False)

def prepare_prompts(traces, codes, note, **kwargs):

        messages = [[
                {"role":"system", "content":judge_prompts()["SYSTEM"]},
                {"role":"user", "content":judge_prompts()["USER"].replace(
                    "$$ADMISSION_NOTE$$", note).replace(
                        "$$TRACE$$", trace).replace(
                            "$$ICD$$", code
                        )}
                
            ] for code, trace in zip(codes, traces)]

        return judge_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False)




def _compute_trace_score(matched_groups: list[list[dict[str, str]]], trace_reward: float) -> float:
    return sum(1 for think_trace, _ in matched_groups if think_trace) * trace_reward

def _compute_model_diagnoses_reward(code: str,
                                    ground_truth: Iterable[str],
                                    score=MATCH_REWARD,
                                    malus=NO_MATCH_MALUS):

    full_match_reward = 0.0
    remaining_ground_truth = {str(x).strip() for x in ground_truth}
    not_fully_matching_codes = []

    
    if code in remaining_ground_truth:
        # Full code match
        full_match_reward += (score*len(ground_truth[0]))
        remaining_ground_truth.remove(code)
    else:
        not_fully_matching_codes.append(code)

    # otherwise check how many digits fit
    partial_overlap, no_matches = compute_partial_digit_overlap_reward(not_fully_matching_codes,
                                                                       remaining_ground_truth)
    score = (full_match_reward
            + partial_overlap * score
            + no_matches * malus)
    return score

def _compute_diagnoses_match_reward(matches: Iterable[dict[str, list[dict]]],
                                    ground_truth: Iterable[str],
                                    score=MATCH_REWARD,
                                    malus=NO_MATCH_MALUS):
    full_match_reward = 0.0
    no_db_hit = 0
    remaining_ground_truth = {str(x).strip() for x in ground_truth}
    not_fully_matching_codes = []

    for match in matches:
        hits = match['hits']
        if len(hits) == 0:  # no hit in database
            no_db_hit += 1
        else:
            predicted_icd_code = hits[0]["icd_code"].strip()
            if predicted_icd_code in remaining_ground_truth:
                # Full code match
                full_match_reward += score
                remaining_ground_truth.remove(predicted_icd_code)
            else:
                not_fully_matching_codes.append(predicted_icd_code)

    # otherwise check how many digits fit
    partial_overlap, no_matches = compute_partial_digit_overlap_reward(not_fully_matching_codes,
                                                                       remaining_ground_truth)
    score = (full_match_reward
            + partial_overlap * score
            + (no_matches + no_db_hit) * malus)
    return score


################################################
###                                          ###
### ---------- Single Think Trace ---------- ###
###                                          ###
################################################


def one_think_trace_total_regex() -> Pattern[AnyStr]:
    return re.compile(
        ANY_OR_NO_WHITESPACE
        + xml_tag_without_inner_tag('think')
        + at_least_once(
            ANY_OR_NO_WHITESPACE
            + xml_tag_without_inner_tag('diagnosis')
        )
        + ANY_OR_NO_WHITESPACE
    )


DIAGNOSIS_REGEX = re.compile(r"<diagnosis>\s*(.*?)\s*</diagnosis>", re.DOTALL)
FORMAT_PATTERN = one_think_trace_total_regex()


def compute_score(solution_str, ground_truth: np.ndarray, method='strict', format_score=FORMAT_REWARD,
                  score=MATCH_REWARD, malus=NO_MATCH_MALUS, **kwargs):
    reward = 0.0

    ### Format Reward:

    completion = "<think>" + solution_str
    try:
        # Check if the format is correct
        match = FORMAT_PATTERN.findall(completion)
        if len(match) == 0:
            reward += 0.0
        else:
            reward += format_score
    except Exception as e:
        logging.warning(f"Exception: {e} Failed to format reward for %s", completion)
        reward += 0.0

    ### Answer Reward:

    # Extract the 'answer' part from the completion
    diagnoses = DIAGNOSIS_REGEX.findall(solution_str)
    if len(diagnoses) == 0:
        return reward

    matches_per_icd_code = msearch_meilisearch(diagnoses)

    matching_codes = []

    ground_truth = {str(x).strip() for x in ground_truth.tolist()}
    for matches in matches_per_icd_code:
        hits = matches['hits']
        if len(hits) > 0:
            matching_icd_code = hits[0]["icd_code"].strip()
            if matching_icd_code in ground_truth:
                # Full code match
                reward += score
                ground_truth.remove(matching_icd_code)
            else:
                matching_codes.append(matching_icd_code)

    # otherwise check how many digits fit
    overlap, no_matches = compute_partial_digit_overlap_reward(matching_codes, ground_truth)
    return reward + overlap * score + no_matches * malus
