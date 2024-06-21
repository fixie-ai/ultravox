import json

import jinja2


def test_quotes():
    with open("tools/ds_tool/soda_alt_last_turn.jinja", "r") as template_file:
        template = template_file.read()

    dialogue = [
        'Have you ever used a double quote (")',
        "Of course, what about a single quote (')?",
        '"Yes, I have."',
        "last turn is ignored!",
    ]

    messages = json.loads(
        jinja2.Template(template).render(dialogue=dialogue, json_dump=json.dumps)
    )
    assert isinstance(messages, list)
    assert all(isinstance(turn, dict) for turn in messages)
    assert messages[-1]["role"] == "user"

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert [x["content"] for x in messages[1:]] == dialogue[:-1]
