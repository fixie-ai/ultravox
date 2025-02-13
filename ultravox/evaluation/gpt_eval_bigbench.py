from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval

BIGBENCH_SYSTEM_PROMPT = """
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.
If the CANDIDATE ANSWER contradicts itself, assess the first proposed answer.
If the CANDIDATE ANSWER provides a final answer and working, assess the final answer only.
If the CANDIDATE ANSWER includes irrelevant information, assess only the relevant information.
If the CANDIDATE ANSWER includes a numeric value it is ok if it is spelled e.g. 7 or seven
It is ok if the CANDIDATE ANSWER involves a misspelling of a person's name e.g. Leda or Lida, Autry or Audrie.
"""

BIGBENCH_USER_PROMPT = """
The question, for reference only: START QUESTION {{ transcript }} \n\nEND QUESTION

The OFFICIAL ANSWER:{{ expected_answer }}

BEGIN CANDIDATE ANSWER TO ASSESS

{{ generated_answer }}

END CANDIDATE ANSWER TO ASSESS

Reply only with CORRECT or INCORRECT.
"""


def evaluate_answer_bigbench(sample: eval_types.Sample) -> eval_types.InstructResult:
    """
    Evaluate an answer using the "CORRECT"/"INCORRECT" prompt style.
    We simply delegate the evaluation to our GPTBasedEvaluator instance's
    evaluate_correct_incorrect method.
    """
    return gpt_eval.gpt_evaluator.evaluate_correct_incorrect(
        BIGBENCH_SYSTEM_PROMPT, BIGBENCH_USER_PROMPT, sample
    )
