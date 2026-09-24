"""Single-turn benchmark tasks with answers that code can check."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jev_router_bench.pool import Tier

GraderName = Literal["exact", "numeric", "set", "code"]


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    intended_tier: Tier  # a hand guess for reporting only; never shown to routers
    prompt: str
    grader: GraderName
    expected: str | float | tuple[str, ...] = ""
    tolerance: float = 1e-6
    tests: str = ""  # hidden asserts for code tasks


ANSWER_SYSTEM_PROMPT = (
    "Solve the task. You may reason briefly first. End your reply with one final line "
    "of the form `ANSWER: <value>` giving only the value, with no units or extra words. "
    "For coding tasks, instead reply with exactly one fenced ```python code block that "
    "defines the requested function, and nothing after it."
)

_CODE = "Reply with one ```python block defining the function."

_CHEAP = (
    Task("extract-email", "extraction", "cheap",
         "Extract the email address from this text: 'Contact Dana at dana.k@example.org or call 555-0199.'",
         "exact", "dana.k@example.org"),
    Task("date-iso", "formatting", "cheap",
         "Convert the date 'March 7, 2024' to ISO 8601 format (YYYY-MM-DD).",
         "exact", "2024-03-07"),
    Task("capital", "lookup", "cheap",
         "What is the capital city of Australia?",
         "exact", "Canberra"),
    Task("add-simple", "arithmetic", "cheap",
         "What is 347 + 589?",
         "numeric", 936.0),
    Task("percent", "arithmetic", "cheap",
         "What is 15% of 240?",
         "numeric", 36.0),
    Task("uppercase", "formatting", "cheap",
         "Convert the text 'hello world' to upper case.",
         "exact", "HELLO WORLD"),
    Task("sort-words", "formatting", "cheap",
         "Sort these words alphabetically and give them comma separated: pear, apple, mango, kiwi",
         "exact", "apple, kiwi, mango, pear"),
    Task("json-field", "extraction", "cheap",
         'Given the JSON {"user": {"name": "Ravi", "age": 31}}, what is the value of user.age?',
         "numeric", 31.0),
    Task("minutes", "arithmetic", "cheap",
         "How many minutes are in 3.5 hours?",
         "numeric", 210.0),
    Task("primes-list", "arithmetic", "cheap",
         "List all prime numbers between 10 and 30, comma separated.",
         "set", ("11", "13", "17", "19", "23", "29")),
)

_MID = (
    Task("train-arrival", "word-problem", "mid",
         "A train leaves at 09:40 and travels 210 km at a constant 84 km/h. "
         "At what time does it arrive? Give HH:MM in 24-hour format.",
         "exact", "12:10"),
    Task("compound-interest", "word-problem", "mid",
         "$2,000 is invested at 5% annual interest, compounded yearly. "
         "What is it worth after 3 years, rounded to 2 decimal places?",
         "numeric", 2315.25, tolerance=0.01),
    Task("letter-count", "string", "mid",
         "How many times does the letter 'r' appear in the phrase 'strawberry raspberry'?",
         "numeric", 6.0),
    Task("reverse-word", "string", "mid",
         "Write the word 'encyclopedia' backwards. Answer with the reversed word only, no spaces "
         "or separators.",
         "exact", "aidepolcycne"),
    Task("ages", "word-problem", "mid",
         "Maria is twice as old as her son. In 12 years she will be 1.5 times as old as him. "
         "How old is Maria now?",
         "numeric", 24.0),
    Task("days-between", "arithmetic", "mid",
         "How many days are there from 2024-01-15 to 2024-03-01 (count the end date, not the start date)?",
         "numeric", 46.0),
    Task("weighted-average", "arithmetic", "mid",
         "Scores: 80 with weight 2, 90 with weight 3, 70 with weight 5. What is the weighted average?",
         "numeric", 78.0),
    Task("code-rle", "code", "mid",
         "Write `run_length_encode(s: str) -> str` that encodes runs of the same character as "
         "the character followed by the run length, e.g. 'aaabcc' -> 'a3b1c2'. "
         "The empty string encodes to ''. " + _CODE,
         "code", tests=(
             'assert run_length_encode("aaabcc") == "a3b1c2"\n'
             'assert run_length_encode("") == ""\n'
             'assert run_length_encode("a") == "a1"\n'
             'assert run_length_encode("aaaaaaaaaaaa") == "a12"\n'
             'assert run_length_encode("abab") == "a1b1a1b1"\n'
         )),
    Task("code-palindrome", "code", "mid",
         "Write `is_palindrome(s: str) -> bool` that returns True when `s` reads the same "
         "forwards and backwards after lower-casing and ignoring every non-alphanumeric "
         "character. " + _CODE,
         "code", tests=(
             'assert is_palindrome("A man, a plan, a canal: Panama") is True\n'
             'assert is_palindrome("race a car") is False\n'
             'assert is_palindrome("") is True\n'
             'assert is_palindrome("0P") is False\n'
             "assert is_palindrome(\"No 'x' in Nixon\") is True\n"
         )),
    Task("code-merge-intervals", "code", "mid",
         "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges "
         "overlapping or touching closed intervals and returns them sorted by start, as a "
         "list of [start, end] lists. The input may be unsorted or empty. " + _CODE,
         "code", tests=(
             "assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]\n"
             "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n"
             "assert merge_intervals([]) == []\n"
             "assert merge_intervals([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]\n"
             "assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]\n"
         )),
)

_STRONG = (
    Task("knights-knaves", "logic", "strong",
         "On an island, knights always tell the truth and knaves always lie. A says: 'B is a knave.' "
         "B says: 'A and C are the same type.' C says nothing. Is C a knight or a knave?",
         "exact", "knave"),
    Task("conditional-dice", "probability", "strong",
         "Two fair six-sided dice are rolled. Given that at least one die shows a 6, what is "
         "the probability that both show 6? Give a fraction in lowest terms.",
         "numeric", 1 / 11, tolerance=1e-4),
    Task("clock-angle", "arithmetic", "strong",
         "What is the smaller angle, in degrees, between the hour and minute hands of an "
         "analog clock at 7:38?",
         "numeric", 1.0, tolerance=1e-6),
    Task("unique-letters", "string", "strong",
         "Take the word 'PROGRAMMING'. Remove every letter that appears more than once in it "
         "(all copies), keeping the remaining letters in their original order. What remains?",
         "exact", "POAIN"),
    Task("digit-sum", "arithmetic", "strong",
         "What is the sum of all the digits of all the integers from 1 to 100 inclusive?",
         "numeric", 901.0),
    Task("four-doors", "probability", "strong",
         "There are 4 closed doors; one hides a car. You pick door 1. The host, who knows where "
         "the car is, opens one of the other doors that hides no car, choosing at random if "
         "several qualify. You then switch to one of the two other unopened doors, chosen "
         "uniformly at random. What is the probability that you win the car?",
         "numeric", 0.375, tolerance=1e-6),
    Task("factorial-zeros", "arithmetic", "strong",
         "How many trailing zeros does 125! (125 factorial) have?",
         "numeric", 31.0),
    Task("code-eval-expr", "code", "strong",
         "Write `evaluate(expr: str) -> int` that evaluates an integer expression with +, -, *, /, "
         "parentheses, unary minus and spaces. Use normal precedence and left associativity. "
         "`/` is integer division that truncates toward zero (so 7/-2 is -3). Do not use eval. "
         + _CODE,
         "code", tests=(
             'assert evaluate("2+3*4") == 14\n'
             'assert evaluate("(1+2)*-3") == -9\n'
             'assert evaluate("7/-2") == -3\n'
             'assert evaluate("-7/2") == -3\n'
             'assert evaluate(" 10 - 2 - 3 ") == 5\n'
             'assert evaluate("2*(3+(4-1))/3") == 4\n'
             'assert evaluate("-(2+3)") == -5\n'
         )),
    Task("code-version-compare", "code", "strong",
         "Write `compare_versions(a: str, b: str) -> int` for dotted numeric versions. Return "
         "1 if a > b, -1 if a < b, 0 if equal. Compare parts numerically, treat missing parts "
         "as 0 and ignore leading zeros (so '1.0' == '1.0.0' and '1.10' > '1.9'). " + _CODE,
         "code", tests=(
             'assert compare_versions("1.0", "1.0.0") == 0\n'
             'assert compare_versions("1.10", "1.9") == 1\n'
             'assert compare_versions("1.0.1", "1") == 1\n'
             'assert compare_versions("0.9", "1.0") == -1\n'
             'assert compare_versions("1.01", "1.001") == 0\n'
             'assert compare_versions("2.0.0.0", "2") == 0\n'
             'assert compare_versions("3.4.5", "3.4.10") == -1\n'
         )),
    Task("code-spiral", "code", "strong",
         "Write `spiral_order(matrix: list[list[int]]) -> list[int]` that returns the elements "
         "of a rectangular matrix in clockwise spiral order starting at the top-left. It must "
         "handle an empty matrix, a single row and a single column. " + _CODE,
         "code", tests=(
             "assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\n"
             "assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == "
             "[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]\n"
             "assert spiral_order([]) == []\n"
             "assert spiral_order([[1], [2], [3]]) == [1, 2, 3]\n"
             "assert spiral_order([[1, 2, 3]]) == [1, 2, 3]\n"
             "assert spiral_order([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 6, 5, 3]\n"
         )),
)

TASKS: tuple[Task, ...] = _CHEAP + _MID + _STRONG


def select_tasks(ids: Sequence[str] | None) -> tuple[Task, ...]:
    if not ids:
        return TASKS
    by_id = {t.id: t for t in TASKS}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise KeyError(
            f"Unknown task(s) {', '.join(unknown)}. Valid: {', '.join(by_id)}"
        )
    return tuple(by_id[i] for i in dict.fromkeys(ids))
