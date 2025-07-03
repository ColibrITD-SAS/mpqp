from abc import ABC, abstractmethod


class Operator(ABC):
    def __init__(self):
        self.key = ""


class BinaryOperator(Operator, ABC):
    def __init__(self):
        self.key = ""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Multiplication(BinaryOperator):
    def __init__(self):
        self.key = "*"

    def __repr__(self) -> str:
        return "Multiplication()"

    def __str__(self) -> str:
        return "*"


class Addition(BinaryOperator):
    def __init__(self):
        self.key = "+"

    def __repr__(self) -> str:
        return "Addition()"

    def __str__(self) -> str:
        return "+"


class Subtraction(BinaryOperator):
    def __init__(self):
        self.key = "-"

    def __repr__(self) -> str:
        return "Subtraction()"

    def __str__(self) -> str:
        return "-"


class UnaryOperator(Operator, ABC):
    def __init__(self):
        self.key = ""


class Minus(UnaryOperator):
    def __init__(self):
        self.key = "-"

    def __repr__(self) -> str:
        return "Minus()"

    def __str__(self) -> str:
        return "-"


class Not(UnaryOperator):
    def __init__(self):
        self.key = "~"

    def __repr__(self) -> str:
        return "Not()"

    def __str__(self) -> str:
        return "~"
