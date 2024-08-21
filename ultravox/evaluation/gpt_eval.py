import dataclasses
from typing import Optional

import jinja2
import openai

from ultravox.evaluation import eval_types

RATING_MODEL = "gpt-4o"
client: Optional[openai.Client] = None


def evaluate_answer_gpt(
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
