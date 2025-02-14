from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval

CONVO_SYSTEM_PROMPT = f"""
You are an expert evaluator of conversational AI systems.
Given a conversation between two parties, the role of the AI system was to follow the flow of the conversation and respond appropriately.
You are given the conversation, the AI model's response, and an exemplary (correct) response.
The AI model response might be truncated, but that should not affect your evaluation.
Your should award 1 point if the model's response is appropriate and follows the conversation, and 0 points if it does not, such as being off-topic or nonsensical.
Your response MUST start with either 0 or 1, followed by a space, and then an explanation for why you awarded that score.
"""

CONVO_USER_PROMPT = """
Using the supplied example of a correct answer, evaluate the model's ability to follow the flow of the conversation in the last message:

Conversation:
{%- for turn in history + [ {"role": "user", "content": question} ] %}
    {% if turn["role"] == "user" %}A{% else %}B{% endif %}: {{ turn["content"] }}
{% endfor %}
    Model (as B): {{ generated_answer }}
    Correct: {{ expected_answer }}
"""


def evaluate_conversation_response(
    sample: eval_types.Sample,
) -> eval_types.InstructResult:
    sample.history = [msg for msg in sample.history if msg["role"] != "system"]
    return gpt_eval.gpt_evaluator.evaluate_binary_with_reason(
        CONVO_SYSTEM_PROMPT, CONVO_USER_PROMPT, sample
    )
