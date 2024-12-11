import json
from typing import Any, Dict, Optional, Set

import jinja2

from ultravox.data import text_proc


def apply_jinja_template(
    template: str, sample: Dict[str, Any], exclude_fields: Optional[Set[str]] = None
):
    """
    Apply a Jinja template to a sample, rendering it into text.
    Jinja template allows for added flexibility as template can include variables and functions.

    Args:
        template: The Jinja template to apply. It can include variables, functions, and control structures.
            Example:
                {{ text }}
                {{ text_proc.format_asr_text(text) }}
        sample: The sample to apply the template to.
        exclude_fields: Fields to exclude from the sample before rendering the template, to avoid loading large fields into memory.
    """
    if exclude_fields:
        # Filter out big fields like audio before the sample is passed into the jinja template
        # otherwise it can lead to unnecessary memory usage.
        sample = {k: sample[k] for k in sample.keys() if k not in exclude_fields}

    try:
        return jinja2.Template(template, undefined=jinja2.StrictUndefined).render(
            **sample, json_dump=json.dumps, text_proc=text_proc
        )
    except jinja2.TemplateError as e:
        print(f"Error rendering template: {e}")
        print(f"template: {template}")
        print(f"sample keys: {list(sample.keys())}, excluded keys: {exclude_fields}")
        raise ValueError(
            f"Template rendering failed. Make sure all keys in the template exist in the sample."
        ) from e
