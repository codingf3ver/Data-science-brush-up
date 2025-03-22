# test_calculator.py

from calculator import add, subtract, multiply, divide
import pytest

def test_add():
    assert add(3, 2) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 2) == 3
    assert subtract(10, 20) == -10

def test_multiply():
    assert multiply(4, 5) == 20
    assert multiply(-1, 2) == -2

def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ValueError):
        divide(10, 0)
