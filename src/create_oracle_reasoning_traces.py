import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, astuple
from pathlib import Path
from typing import Iterable
import torch
import datasets
import torch
from datasets import Dataset, tqdm
from torch.utils.data import DataLoader

from transformers import HfArgumentParser
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

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
    output_dir: Path = Path("output")
    tokenizer_name: str = None
    prompt_column: str = "messages"
    batch_size: int = 8
    num_gpus: int = torch.cuda.device_count()
    min_max_tokens: int = 1024
    dtype: str = 'bfloat16'


@torch.inference_mode()
def main(args: ScriptArguments):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data: %s", args.data)

    dataset = Dataset.from_parquet(str(args.data))

    #torch._dynamo.disable()
    logger.info("Loading model: %s", args.model_name)
    llm = LLM(model=args.model_name,
              enable_prefix_caching=True,
              tensor_parallel_size=args.num_gpus,
              dtype=args.dtype,
              download_dir=os.getenv('HF_HOME', None),
              gpu_memory_utilization=0.8
              )
    set_llm_to_min_max_tokens_or_default(args, llm)
    sampling_params = SamplingParams(n=1,
                                     temperature=0.9,
                                     seed=42,
                                     max_tokens=1024)
    logger.info("Starting evaluation")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=device == 'cuda',
                            collate_fn=make_collate_fn(
                                tokenizer=llm.get_tokenizer(),
                                prompt_column=args.prompt_column,
                            ))

    outputs = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    
        completions = unpack_vllm_output(llm.generate(prompts=batch['prompts'], sampling_params=sampling_params))
        outputs.append(completions)
    outputs = ["<think>" + x for xs in outputs for x in xs]
    dataset = dataset.add_column("oracle_traces",outputs) 
    output_file = args.output_dir / (args.data.stem + ".parquet") 
    logger.info('Saving results to %s', output_file)
    dataset.to_parquet(output_file)

def set_llm_to_min_max_tokens_or_default(args, llm):
    max_tokens = llm.get_default_sampling_params().max_tokens
    if max_tokens < args.min_max_tokens:
        logger.warning(f'Default max_tokens only at {max_tokens} tokens, setting to {args.min_max_tokens}')
    llm.default_sampling_params['max_tokens'] = args.min_max_tokens


def make_collate_fn(tokenizer, prompt_column):

    def collate_fn(example):
        prompts = [x["oracle_prompts"] for x in example]
        hadm_ids = [x["hadm_id"] for x in example]
        codes = [x["first_code"] for x in example]

        return {'prompts': prompts,
                "hadm_ids": hadm_ids,
                "codes": codes}

    return collate_fn

def verify_and_pack_output(outputs, codes):
    pass
def unpack_vllm_output(outputs: list[RequestOutput]) -> list[str]:
    return [output.outputs[0].text for output in outputs]




if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()
    logging.info(f"Script arguments: {script_args}")
    main(script_args)
