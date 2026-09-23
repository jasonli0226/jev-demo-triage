import pytest

from jev_router_bench.llm_choice import LlmRouteError, parse_choice


def test_plain_json_with_probabilities():
    text = '{"choice": "mid", "probabilities": {"cheap": 0.2, "mid": 0.7, "strong": 0.1}}'
    assert parse_choice(text) == ("mid", {"cheap": 0.2, "mid": 0.7, "strong": 0.1})


def test_choice_without_probabilities():
    assert parse_choice('```json\n{"choice": "cheap"}\n```') == ("cheap", None)


def test_json_inside_prose():
    assert parse_choice('Sure. {"choice": "strong"} done')[0] == "strong"


def test_integer_probabilities_are_accepted():
    assert parse_choice('{"choice": "strong", "probabilities": {"strong": 1}}') == (
        "strong", {"strong": 1.0}
    )


def test_tie_including_choice_is_accepted():
    text = '{"choice": "mid", "probabilities": {"cheap": 0.5, "mid": 0.5}}'
    assert parse_choice(text)[0] == "mid"


@pytest.mark.parametrize(
    "text",
    [
        "mid",
        '{"tier": "mid"}',
        '{"choice": "huge"}',
        '{"choice": 2}',
        '{"choice": "mid", "probabilities": {"mid": 1.5}}',
        '{"choice": "mid", "probabilities": {"mid": true}}',
        '{"choice": "mid", "probabilities": {"giant": 0.9}}',
        '{"choice": "mid", "probabilities": {"cheap": 0.8, "mid": 0.2}}',
        '{"choice": "mid", "probabilities": {}}',
        '{"choice": "mid", "probabilities": [0.1]}',
        '{"choice": "mid", "choice": "cheap"}',
        '{"choice": "mid"} {"choice": "cheap"}',
    ],
)
def test_invalid_answers_raise(text):
    with pytest.raises(LlmRouteError):
        parse_choice(text)


def test_agreeing_duplicates_are_accepted():
    assert parse_choice('{"choice": "mid"} again {"choice": "mid"}') == ("mid", None)
