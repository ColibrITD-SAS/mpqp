from abc import ABC, abstractmethod


class Operator(ABC):
    """Abstract class used to define operators (in the arithmetic sense) used by oter classes
    to create expressions with editable variables (see :class:`~mpqp.execution.vqa.qubo.Qubo`).
    """


class BinaryOperator(Operator, ABC):
    """Abstract class used to create a binary operator."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Multiplication(BinaryOperator):
    """Class used to create an arithmetic multiplication operation."""

    def __repr__(self) -> str:
        return "Multiplication()"

    def __str__(self) -> str:
        return "*"


class Addition(BinaryOperator):
    """Class used to create an arithmetic addition operation."""

    def __init__(self):
        self.key = "+"

    def __repr__(self) -> str:
        return "Addition()"

    def __str__(self) -> str:
        return "+"


class Subtraction(BinaryOperator):
    """Class used to create an arithmetic subtraction operation."""

    def __init__(self):
        self.key = "-"

    def __repr__(self) -> str:
        return "Subtraction()"

    def __str__(self) -> str:
        return "-"


class UnaryOperator(Operator, ABC):
    """Abstract class used to create unary operators."""

    def __init__(self):
        self.key: str


class Minus(UnaryOperator):
    r"""Class used to inverse the sign of a Qubo expression `q \rightarrow -q`."""

    def __init__(self):
        self.key = "-"

    def __repr__(self) -> str:
        return "Minus()"

    def __str__(self) -> str:
        return "-"


class Not(UnaryOperator):
    """Class used to create the boolean operator not."""

    def __init__(self):
        self.key = "~"

    def __repr__(self) -> str:
        return "Not()"

    def __str__(self) -> str:
        return "~"
