"""This module provides functionalities for working with decision trees, 
allowing for seamless navigation and interaction within a tree structure.

You can define a :class:`QuestionNode`, containing your question and options.
Each option (:class:`AnswerNode`) contains the description of the option
together with optional actions and follow up question.

To run your choice tree, just run :func:`run_choice_tree` on your root question.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TypeVar, TYPE_CHECKING

from pick import pick
from typeguard import typechecked

from mpqp.tools.generics import find

T = TypeVar("T")


@dataclass
class AnswerNode:
    """Represents a node in a decision tree corresponding to an answer to a
    question. An answer can lead to an action, or to another question.

    Args:
        label: See attribute description.
        action: See attribute description.
        next_question: See attribute description.
    """

    label: str
    """The label or text associated with the answer."""
    action: Callable[..., tuple[str, Iterable[Any]]]
    """A callable representing an action function linked to an answer. The
    return value of the action is composed of the text to display after it ran 
    (something like "Success", "Failure", or "You're a wonderful human being"),
    and the parameters to pass to the next action to be executed. These
    parameters to be passed between actions are useful to keep memory of the
    previous actions for instance."""
    next_question: Optional["QuestionNode"] = None
    """An optional reference to the next question node."""


@dataclass
class QuestionNode:
    """Represents a node in a decision tree corresponding to a question.

    Args:
        label: The label or text associated with the question.
        answers: List of possible answers to this question.
        leaf_loop_to_here: If ``True`` answers without followup questions will
            loop back to here.
    """

    label: str
    answers: list[AnswerNode]
    leaf_loop_to_here: bool = False

    def __post_init__(self):
        to_visit: list[QuestionNode] = [self]
        visited: list[QuestionNode] = []
        while len(to_visit) != 0:
            for q_index, ques in enumerate(to_visit):
                if ques not in visited:
                    for ans in ques.answers:
                        if ans.next_question is None:
                            ans.next_question = self
                        else:
                            visited.append(ans.next_question)
                    visited.append(ques)
                    del to_visit[q_index]


@typechecked
def run_choice_tree(question: QuestionNode):
    """Execute the choice tree by starting with the question node in parameter.

    Args:
        question: Root question from which the choice tree will start
    """
    prev_args = []
    prev_message = ""
    next_question = question

    # TODO: add the following to pick once my PR is merged
    # KEY_CTRL_C = 3
    # KEY_ESCAPE = 27
    # KEYS_QUIT = (KEY_CTRL_C, KEY_ESCAPE, ord("q"))

    try:
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

    except KeyboardInterrupt:
        print()
        pass


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

    if TYPE_CHECKING:
        assert choice_tree.answers[-1].next_question is not None

    choice_tree.answers[-1].next_question.answers[0].next_question = choice_tree

    run_choice_tree(choice_tree)
