Tools
=====

Some additional tools are provided with our library. Even though most of them
are geared at internal usage, they are all presented here. Amongst them, the
ones most probable of being of use for you are in:

- the section :ref:`viz` presents visualization tools for several data types.
  They might be integrated in these types if they are popular enough.
- the section :ref:`math` presents mathematical tools for linear algebra,
  functions generalized to more data types, etc...

.. _viz:

Visualization
-------------

.. code-block::python

    from mpqp.tools.visualization import *

.. automodule:: mpqp.tools.visualization

.. _math:

Useful Maths Operations
-----------------------

.. code-block::python

    from mpqp.tools.maths import *

.. automodule:: mpqp.tools.maths

Generics
--------

We centralize generic types and associated or general recurrent manipulations in
a single module.

.. automodule:: mpqp.tools.generics

Errors
------

In order to provide more explicit messages concerning errors that the user can
encounter while using the library, we introduce custom exceptions. When it is
relevant, we also append the trace of the error raised by a provider's SDK.

.. automodule:: mpqp.tools.errors

Choice Tree
-----------

This module provides functionalities for working with decision trees, allowing for
seamless navigation and interaction within a tree structure.

The user defines a :class:`QuestionNode<mpqp.tools.choice_tree.QuestionNode>`,
containing a question or requesting and answer from the user. Then the user can
select among a list of possible answers, encoded in
:class:`AnswerNode<mpqp.tools.choice_tree.AnswerNode>`. For each answer, either
another question will follow, or an action can be executed.


.. autoclass:: mpqp.tools.choice_tree.AnswerNode

.. autoclass:: mpqp.tools.choice_tree.QuestionNode

To execute the choice tree in a console mode, one has to execute the function
:func:`run_choice_tree<mpqp.tools.choice_tree.run_choice_tree>` while giving the
root ``QuestionNode`` in the choice tree.

.. autofunction:: mpqp.tools.choice_tree.run_choice_tree

Example
^^^^^^^

.. code-block:: python

    def tea_picker(*_):
        name = input("What's your favorite tea brand?\n\t")
        return f"Very good choice, I love {name}!", [name]

    choice_tree = QuestionNode(
        "Hey, what's your favorite beverage?",
        [
            AnswerNode(
                "Tea",
                tea_picker,
                QuestionNode(
                    "Do you want to come at my place to drink some",
                    [
                        AnswerNode(
                            "Yes",  lambda n: (f"Cool, let's go for this cup of {n} :D", [])
                        ),
                        AnswerNode("Can't :(", lambda n: ("No worries, another time!", [])),
                    ],
                ),
            ),
            AnswerNode(
                "Coffee",
                lambda: ("I get it, ou gotta do what you gota do to stay awake", []),
                QuestionNode(
                    "And how do you like it?",
                    [
                        AnswerNode(
                            "American style", 
                            lambda: (
                                "Hydrating is important, but I'm not sure this counts as coffee", 
                                [],
                            ),
                        ),
                        AnswerNode("Italian", lambda: ("The only right way!", [])),
                    ],
                ),
            ),
        ],
    )

    assert choice_tree.answers[-1].next_question is not None

    choice_tree.answers[-1].next_question.answers[0].next_question = choice_tree

    run_choice_tree(choice_tree)