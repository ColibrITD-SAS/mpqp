from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TypeVar
from pick import pick
from typeguard import typechecked

from mpqp.tools.generics import find

T = TypeVar("T")


@dataclass
class AnswerNode:
    """Represents a node in a decision tree corresponding to an answer to a question. An answer can lead to an action,
    or to another question.

    Args:
        label: See attribute description.
        action: See attribute description.
        next_question: See attribute description.
    """

    label: str
    """The label or text associated with the answer."""
    action: Callable[..., tuple[str, Iterable[Any]]]
    """A callable representing an action function linked to an answer."""
    next_question: Optional["QuestionNode"] = None
    """An optional reference to the next question node."""


@dataclass
class QuestionNode:
    """Represents a node in a decision tree corresponding to a question.

    Args:
        label: See attribute description.
        answers: See attribute description.
    """
    label: str
    """The label or text associated with the question."""
    answers: list[AnswerNode]
    """List of possible answers to this question."""


@typechecked
def run_choice_tree(question: QuestionNode):
    """
    Execute the choice tree by starting with the question node in parameter.

    Args:
        question: Root question from which the choice tree will start
    """
    # 3M-TODO: allow exit with Ctrl+C and Q
    prev_args = []
    prev_message = ""
    next_question = question
    while True:
        option, _ = pick(
            list(map(lambda a: a.label, next_question.answers)) + ["Exit"],
            prev_message + "\n\n" + next_question.label,
            indicator="=>",
        )
        if option == "Exit":
            return
        selected_answer = find(next_question.answers, lambda a: a.label == option)
        prev_message, prev_args = selected_answer.action(*prev_args)
        next_question = selected_answer.next_question
        if next_question is None:
            pick(["Press 'Enter' to continue"], prev_message, indicator="")
            return


# Example:
if __name__ == "__main__":

    def date_name_picker():
        name = input("What's your name?\n\t")
        return f"Then it's a date {name}!", [name]

    choice_tree = QuestionNode(
        "Are you free?",
        [
            AnswerNode(
                "Yes",
                date_name_picker,
                QuestionNode(
                    "Do you have a preference for the day?",
                    [
                        AnswerNode(
                            "Not at all", lambda n: (f"Tomorrow then {n} :D", [])
                        ),
                        AnswerNode("Tomorrow", lambda n: (f"Tomorrow then {n} :D", [])),
                    ],
                ),
            ),
            AnswerNode(
                "No",
                lambda: ("What a shame :(", []),
                QuestionNode(
                    "Would you be willing to reconsider ?",
                    [
                        AnswerNode("Yes", lambda: ("Cool!", [])),
                        AnswerNode("No", lambda: ("Sad :'(", [])),
                    ],
                ),
            ),
        ],
    )

    assert choice_tree.answers[-1].next_question is not None

    choice_tree.answers[-1].next_question.answers[0].next_question = choice_tree

    run_choice_tree(choice_tree)
