Tools
=====

Some additional tools are provided with our library. Even though most of them
are geared at internal usage, they are all presented here. Amongst them, the
ones most probable of being of use for you are in the :ref:`math` section,
presenting mathematical tools for linear algebra, functions generalized to more 
data types, etc...

.. _math:

Useful Maths Operations
-----------------------

.. code-block:: python
    :class: import

    from mpqp.tools.maths import *

.. automodule:: mpqp.tools.maths

Generics
--------

.. code-block:: python
    :class: import

    from mpqp.tools.generics import *

.. automodule:: mpqp.tools.generics

Display helpers
---------------

.. code-block:: python
    :class: import

    from mpqp.tools.display import *

.. automodule:: mpqp.tools.display

Errors
------

.. code-block:: python
    :class: import

    from mpqp.tools.errors import *

.. automodule:: mpqp.tools.errors

Circuit tricks
--------------

.. code-block:: python
    :class: import

    from mpqp.tools.circuit import *

.. automodule:: mpqp.tools.circuit

Choice Tree
-----------

.. code-block:: python
    :class: import

    from mpqp.tools.choice_tree import *

.. automodule:: mpqp.tools.choice_tree

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
                lambda: ("I get it, you gotta do what you gotta do to stay awake", []),
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

Theoretical simulations
-----------------------

.. code-block:: python
    :class: import

    from mpqp.tools.theoretical_simulation import *

.. automodule:: mpqp.tools.theoretical_simulation