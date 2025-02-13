import dataclasses
from typing import Optional

import jinja2
import openai

from ultravox.evaluation import eval_types


class GPTBasedEvaluator:
    """
    Encapsulates GPT-based evaluation logic. This class holds a single client
    instance, runs inference as needed, and provides different evaluation
    methods with different rating logics.
    """

    def __init__(self, rating_model: str = "gpt-4o"):
        self.rating_model = rating_model
        self.client: Optional[openai.Client] = None

    def _get_client(self) -> openai.Client:
        """
        Returns a cached openai.Client instance, creates it if needed.
        """
        if self.client is None:
            self.client = openai.Client()
        return self.client

    def _run_gpt_inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        sample: eval_types.Sample,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Common utility function that runs OpenAI Chat Completions and returns
        the raw response text.
        """
        template = jinja2.Template(user_prompt)

        response = self._get_client().chat.completions.create(
            model=self.rating_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": template.render(**dataclasses.asdict(sample)),
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        rating_text = response.choices[0].message.content
        assert rating_text is not None
        return rating_text

    def evaluate_binary_with_reason(
        self,
        sys_prompt: str,
        user_prompt: str,
        sample: eval_types.Sample,
        rating_at_end: bool = False,
    ) -> eval_types.InstructResult:
        """
        A GPT-based evaluation that expects the model to return a line
        beginning with either 0 or 1 (and an optional explanation).
        """
        rating_text = self._run_gpt_inference(
            sys_prompt, user_prompt, sample, max_tokens=1024, temperature=0.7
        )
        assert rating_text is not None
        score = 0
        reason = ""
        try:
            # Depending on the prompt, the rating may be at the beginning or the end
            if rating_at_end:
                rating = int(rating_text.split()[-1])
                reason = rating_text[: -len(str(rating))].strip()
            else:
                rating = int(rating_text.strip()[0])
                reason = rating_text[2:].strip() if len(rating_text) > 2 else ""
            if rating in (0, 1):
                score = rating
        except (ValueError, IndexError):
            pass

        return eval_types.InstructResult(score=score, reason=reason)

    def evaluate_correct_incorrect(
        self, sys_prompt: str, user_prompt: str, sample: eval_types.Sample
    ) -> eval_types.InstructResult:
        """
        A GPT-based evaluation that expects the model to return 'CORRECT'
        or 'INCORRECT' (or something else). If it returns exactly 'CORRECT',
        the score is 1; otherwise 0.
        """
        rating_text = self._run_gpt_inference(sys_prompt, user_prompt, sample)
        assert rating_text is not None
        score = 1 if rating_text == "CORRECT" else 0
        return eval_types.InstructResult(score=score, reason="")

    def evaluate_score_scalar(
        self, sys_prompt: str, user_prompt: str, sample: eval_types.Sample
    ) -> eval_types.InstructResult:
        """
        A GPT-based evaluation that expects the model to return a scalar score.
        """
        rating_text = self._run_gpt_inference(
            sys_prompt, user_prompt, sample, max_tokens=1024, temperature=0.7
        )
        assert rating_text is not None
        try:
            score = int(rating_text.split()[-1])
            reason = rating_text[: -len(str(score))].strip()
        except:
            score = 0
            reason = ""
        return eval_types.InstructResult(score=score, reason=reason)


gpt_evaluator = GPTBasedEvaluator()
