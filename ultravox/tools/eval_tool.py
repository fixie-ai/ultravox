import argparse
import dataclasses
from typing import IO

import simple_parsing

from ultravox.evaluation import eval
from ultravox.evaluation import eval_types


@dataclasses.dataclass
class EvalArgs:
    # Path to the audio file
    file: IO = simple_parsing.field(type=argparse.FileType("rb"), alias="-f")
    # Metric to use for evaluation (e.g., "asr" or "boolq")
    metric: str = simple_parsing.field(default="asr", alias="-m")
    # Verbose output
    verbose: bool = simple_parsing.field(default=False, alias="-v")


def main(args: EvalArgs):
    scores = []
    for i, line in enumerate(args.file.readlines()):
        sample = eval_types.Sample.from_json(line)
        result = eval.evaluate_answer(sample, metric=args.metric)
        assert result.score is not None, "Rating failed."
        scores.append(result.score)
        average = sum(scores) / len(scores)
        print(f"{i}: score={result.score:.2f} average={average:.2f}")
        if args.verbose and isinstance(result, eval_types.InstructResult):
            print(f"  reason={result.reason}")


if __name__ == "__main__":
    main(simple_parsing.parse(EvalArgs))
