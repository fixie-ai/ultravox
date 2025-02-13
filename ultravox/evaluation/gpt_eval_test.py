import re
from unittest import mock

from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval
from ultravox.evaluation import gpt_eval_conv


def test_evaluate_conversation():
    gpt_eval.gpt_evaluator.client = mock.MagicMock()
    sample = eval_types.Sample(
        index=0,
        history=[
            {"role": "system", "content": "Blah blah blah"},
            {"role": "user", "content": "T1"},
            {"role": "assistant", "content": "T2"},
        ],
        question="T3",
        generated_answer="T4",
        expected_answer="EXP",
        transcript="",
    )
    expected_turns = "A: T1\n\nB: T2\n\nA: T3\n\nModel (as B): T4\nCorrect: EXP"

    gpt_eval_conv.evaluate_conversation_response(sample)

    completion_args = gpt_eval.gpt_evaluator.client.chat.completions.create.call_args[1]
    assert len(completion_args["messages"]) == 2
    assert completion_args["messages"][0]["role"] == "system"
    assert completion_args["messages"][1]["role"] == "user"
    gpt_question = re.sub("\n *", "\n", completion_args["messages"][1]["content"])
    assert expected_turns in gpt_question
    assert "Blah blah blah" not in gpt_question
