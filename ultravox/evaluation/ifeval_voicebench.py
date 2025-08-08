"""Binary of evaluating instruction following. See README.md."""

import abc
import dataclasses
from typing import Dict, Optional, Union

import numpy as np

from ultravox.evaluation import eval_types
from ultravox.evaluation.ifeval import instructions_registry


@dataclasses.dataclass
class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, data):
        pass

    @abc.abstractmethod
    def evaluate_strict(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_loose(self, data):
        raise NotImplementedError


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def read_prompt_list(sample: eval_types.Sample):
    """Read input`s from data."""
    inputs = []

    if sample.extra_kwargs is None:
        raise ValueError("extra_kwargs must be provided")

    extra = sample.extra_kwargs

    # Check required keys exist
    required_keys = ["key", "instruction_id_list", "kwargs"]
    missing = [k for k in required_keys if k not in extra]
    if missing:
        raise ValueError(
            f"extra_kwargs missing required fields: {missing}. "
            "Expected keys: ['key', 'instruction_id_list', 'kwargs']"
        )

    # key -> int
    key = extra.get("key")
    if isinstance(key, (int, str)):
        key = int(key)
    else:
        raise TypeError(f"extra_kwargs['key'] must be an int or castable to int: {key}")

    # instruction_id_list -> list[str]
    instr_raw = extra.get("instruction_id_list")
    if not isinstance(instr_raw, list):
        raise TypeError("extra_kwargs['instruction_id_list'] must be a list")
    instruction_id_list = [str(x) for x in instr_raw]

    # kwargs -> list[Dict[str, Optional[Union[str, int]]]]
    kwargs_raw = extra.get("kwargs")
    if not isinstance(kwargs_raw, list):
        raise TypeError("extra_kwargs['kwargs'] must be a list of dictionaries")
    kwargs: list[Dict[str, Optional[Union[str, int]]]] = []
    for item_raw in kwargs_raw:
        if not isinstance(item_raw, dict):
            raise TypeError(
                "Each element of extra_kwargs['kwargs'] must be a dictionary"
            )
        item: Dict[str, Optional[Union[str, int]]] = {}
        for k, v in item_raw.items():
            key_str = str(k)
            if isinstance(v, (str, int)):
                item[key_str] = v
            else:
                raise TypeError(
                    f"extra_kwargs['kwargs'] must be a list of dictionaries with string or int values: {item_raw}"
                )
        kwargs.append(item)

    if len(kwargs) != len(instruction_id_list):
        raise ValueError(
            "Length mismatch: instruction_id_list and kwargs must have the same length"
        )

    inputs.append(
        InputExample(
            key=key,
            instruction_id_list=instruction_id_list,
            prompt=sample.transcript,
            kwargs=kwargs,
        )
    )
    return inputs


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def read_prompt_to_response_dict(sample: eval_types.Sample):
    """Creates dictionary matching prompt and response."""
    return_dict = {}
    if isinstance(sample, list):
        sample = sample[0]
    tmp = sample.generated_answer
    if tmp.startswith("<1>") or tmp.startswith("<2>") or tmp.startswith("<3>"):
        tmp = tmp[3:].strip()
    if tmp.endswith("<|user|>"):
        tmp = tmp[:-8].strip()

    return_dict[sample.transcript] = tmp
    return return_dict


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)
    return {
        "prompt": prompt_correct / prompt_total,
        "instruction": instruction_correct / instruction_total,
    }


class IFEvaluator(Evaluator):
    @staticmethod
    def instruction_following_evaluate(
        sample: eval_types.Sample,
    ) -> eval_types.InstructResult:
        inputs = read_prompt_list(sample)
        prompt_to_response = read_prompt_to_response_dict(sample)
        # get instruction following results
        results = {}
        outputs = []
        for inp in inputs:
            outputs.append(test_instruction_following_strict(inp, prompt_to_response))
        for key, value in print_report(outputs).items():
            results[f"strict-{key}"] = value
        outputs = []
        for inp in inputs:
            outputs.append(test_instruction_following_loose(inp, prompt_to_response))
        for key, value in print_report(outputs).items():
            results[f"loose-{key}"] = value
        results["final"] = np.mean(list(results.values()))
        return eval_types.InstructResult(
            score=np.mean(list(results.values())), reason=""
        )
