import pytest

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.noise import Depolarizing, NoiseModel

def f():
    assert True