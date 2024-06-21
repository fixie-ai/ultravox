import dataclasses
from typing import Optional

import jinja2
import openai

from ultravox.evaluation import eval_types

INSTRUCT_SYSTEM_PROMPT = f"""
You are an expert evaluator of AI systems.
Given a question with a specified instruction, you will be rating the correctness of an AI model's ability to follow that instruction.
Based on the supplied answer, and exemplary (correct) answer, you will rate the model's answer as either correct or incorrect.
Award 1 point if the model followed the instruction, and 0 points if it did not.
For example, given a question with an instruction of "Write a sentence about pickleball",
- if the model responds "Pickleball is a tennis-like game played with a wiffle ball.", you should award 1 point.
- if the model responds "Pickleball is a type of fruit", you should award 0 points.
- if the model responds with something off-topic or nonsensical, you should award 0 points.
Your response MUST start with either 0 or 1, followed by a space, and then an explanation for why you awarded that score.
"""
INSTRUCT_USER_PROMPT = """
Using the supplied correct answer as an example, evaluate the model's ability to follow the instructions in the question below:
Question: {{ question }}
Model answer: {{ generated_answer }}
Correct answer: {{ expected_answer }}
"""


BOOLQ_SYSTEM_PROMPT = f"""
You are an expert evaluator of AI systems.
Given a question with a known true/false answer, you will be rating the correctness of an AI model's answer to that same question.
Based on the supplied question, answer, and expected (correct) answer, you will rate the model's answer as either correct or incorrect.
Award 1 point if the model's answer matches the correct answer, and 0 points if the model's answer does not match, or cannot be converted to a true/false verdict.
Model answers of the form "True", "Yes", "Yeah", etc., should be considered to match a True answer.
Model answers of the form "False", "No", "Incorrect", etc., should be considered to match a False answer.
Only use the supplied correct answer to make your decision; DO NOT use your own knowledge to determine correctness.
Your response MUST start with either 0 or 1, followed by a space, and then a brief explanation for why you awarded that score.
"""
BOOLQ_USER_PROMPT = """
Using the supplied correct answer as ground truth, evaluate the model's answer to the question below:
Question: {{ question }}
Model answer: {{ generated_answer }}
Correct answer: {{ expected_answer }}
"""


CONVO_SYSTEM_PROMPT = f"""
You are an expert evaluator of conversational AI systems.
Given a conversation between two parties, the role of the AI system was to follow the flow of the conversation and respond appropriately.
You are given the conversation, the AI model's response, and an exemplary (correct) response.
Your should award 1 point if the model's response is appropriate and follows the conversation, and 0 points if it does not, such as being off-topic or nonsensical.
Your response MUST start with either 0 or 1, followed by a space, and then an explanation for why you awarded that score.
"""

CONVO_USER_PROMPT = """
Using the supplied example of a correct answer, evaluate the model's ability to follow the flow of the conversation in the last message:

Conversation:
{%- for turn in history + [ question ] %}
    {{ loop.cycle('A', 'B') }}: {{ turn }}
{% endfor %}
    Model (as {% if history | length is odd %}A{% else %}B{% endif %}): {{ generated_answer }}
    Correct: {{ expected_answer }}
"""

RATING_MODEL = "gpt-4o"
client: Optional[openai.Client] = None


def _evaluate_answer_gpt(
    sys_prompt: str, user_prompt: str, sample: eval_types.Sample
) -> eval_types.InstructResult:
    global client
    if client is None:
        client = openai.Client()
    template = jinja2.Template(user_prompt)
    response = client.chat.completions.create(
        model=RATING_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": template.render(**dataclasses.asdict(sample)),
            },
        ],
        max_tokens=50,
        temperature=0,
    )
    rating_text = response.choices[0].message.content
    assert rating_text is not None
    score = None
    try:
        rating = int(rating_text.strip()[0])
        if rating == 0 or rating == 1:
            score = rating
    except ValueError:
        pass

    return eval_types.InstructResult(score=score, reason=rating_text[2:])


def evaluate_answer_boolq(sample: eval_types.Sample) -> eval_types.InstructResult:
    return _evaluate_answer_gpt(BOOLQ_SYSTEM_PROMPT, BOOLQ_USER_PROMPT, sample)


def evaluate_answer_instruct(sample: eval_types.Sample) -> eval_types.InstructResult:
    return _evaluate_answer_gpt(INSTRUCT_SYSTEM_PROMPT, INSTRUCT_USER_PROMPT, sample)


def evaluate_conversation_response(
    sample: eval_types.Sample,
) -> eval_types.InstructResult:
    return _evaluate_answer_gpt(CONVO_SYSTEM_PROMPT, CONVO_USER_PROMPT, sample)
