#!/usr/bin/env python
import dataclasses
import json
import sys
from pathlib import Path
from typing import List, Dict

import simple_parsing

from ultravox.evaluation import eval_types
from ultravox.evaluation.eval import evaluate_answers
from ultravox.utils import string_helpers

@dataclasses.dataclass
class ScoreArgs:
    # Path to JSON file containing samples
    input: Path = simple_parsing.field(alias="-i")
    # Metric to use for evaluation
    metric: str = simple_parsing.field(alias="-m")
    # Additional arguments for the metric configuration
    args: Dict[str, str] = simple_parsing.field(alias="-a", default_factory=dict)

def main():
    args = simple_parsing.parse(
        config_class=ScoreArgs,
        args=[string_helpers.fix_hyphens(arg) for arg in sys.argv[1:]],
    )

    # Load samples from JSON file
    with open(args.input, 'r') as f:
        samples_data = json.load(f)
    
    samples = [eval_types.Sample(**sample) for sample in samples_data]
    
    # Create EvalConfig with args
    metric_config = eval_types.EvalConfig(metric=args.metric, args=args.args)
    
    # Evaluate samples
    result = evaluate_answers(samples, metric_config)
    
    # Print result
    print(f"input: {args.input}")
    print(f"args: {args.args}")
    print(f"{args.metric}: {result.score:.2f}")

if __name__ == "__main__":
    main()