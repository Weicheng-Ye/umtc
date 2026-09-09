"""A small mathematical-expression grammar for JSON symbol functions.

Expressions support numeric and string literals, argument names, ``pi``, direct argument
integer subscripts, integer tuple/list literals, arithmetic and bitwise
operators, comparisons, ``and``/``or``/``not``, and conditional expressions.
The only calls are ``sqrt``, ``exp``, ``sin``, ``cos``, ``conj``, ``abs``, and
``complex``. This module interprets the validated syntax tree; it does not
execute Python source. Python functions supplied through the Python API are
separate from this deliberately limited JSON grammar.
"""

from __future__ import annotations

import ast
import cmath
import keyword
import math
import operator
from typing import Any, Callable

_MAX_LENGTH = 4096
_MAX_NODES = 256
_MAX_DEPTH = 32
_MAX_SEQUENCE = 256
_MAX_MAGNITUDE = 1e100
_MAX_EXPONENT = 1024

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitXor: operator.xor, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_CALLS = {
    "sqrt": (cmath.sqrt, (1,)), "exp": (cmath.exp, (1,)),
    "sin": (cmath.sin, (1,)), "cos": (cmath.cos, (1,)),
    "conj": (lambda value: complex(value).conjugate(), (1,)),
    "abs": (abs, (1,)), "complex": (complex, (1, 2)),
}


def _number(value: Any) -> int | float | complex:
    if type(value) not in (int, float, complex):
        raise ValueError("Expression arithmetic requires numeric scalars")
    # Checking an integer's size first also avoids an overflowing conversion.
    if type(value) is int:
        finite = value.bit_length() <= 333
    else:
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
    if not finite or abs(value) > _MAX_MAGNITUDE:
        raise ValueError("Expression numeric values must be finite and at most 1e100 in magnitude")
    return value


def _integer_sequence(values: Any) -> None:
    if len(values) > _MAX_SEQUENCE or any(type(value) is not int for value in values):
        raise ValueError("Expression tuple/list values must contain at most 256 integers")
    for value in values:
        _number(value)


def _argument(value: Any) -> None:
    if type(value) in (tuple, list):
        _integer_sequence(value)
    elif type(value) is str:
        if len(value) > _MAX_LENGTH:
            raise ValueError("Expression string arguments must have at most 4096 characters")
    else:
        _number(value)


def _index(node: ast.AST) -> int:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise ValueError("Expression subscripts must be integer literals")
    return sign * _number(node.value)


def _validate(node: ast.AST, arguments: set[str], depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("Expression nesting exceeds 32 levels")
    children: list[ast.AST]
    if isinstance(node, ast.Constant):
        if type(node.value) is str:
            _argument(node.value)
        else:
            _number(node.value)
        children = []
    elif isinstance(node, ast.Name):
        if node.id not in arguments and node.id != "pi":
            raise ValueError(f"Unknown expression name {node.id!r}")
        children = []
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name) or node.value.id not in arguments:
            raise ValueError("Expression subscripts may only index arguments directly")
        _index(node.slice)
        children = []
    elif isinstance(node, (ast.Tuple, ast.List)):
        children = node.elts
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        children = [node.left, node.right]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        children = [node.operand]
    elif isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
        children = [node.left, *node.comparators]
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        children = node.values
    elif isinstance(node, ast.IfExp):
        children = [node.test, node.body, node.orelse]
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _CALLS:
            raise ValueError("Expression calls must use a supported mathematical function")
        if node.keywords or len(node.args) not in _CALLS[node.func.id][1]:
            raise ValueError(f"Invalid arguments to expression function {node.func.id!r}")
        children = node.args
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
    for child in children:
        _validate(child, arguments, depth + 1)


def _interpret(node: ast.AST, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return math.pi if node.id == "pi" else values[node.id]
    if isinstance(node, ast.Subscript):
        value = values[node.value.id]
        if type(value) not in (tuple, list):
            raise ValueError("Expression subscripts require tuple/list arguments")
        return value[_index(node.slice)]
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_interpret(item, values) for item in node.elts]
        _integer_sequence(items)
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.BinOp):
        left = _number(_interpret(node.left, values))
        right = _number(_interpret(node.right, values))
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Expression exponent magnitude must be at most 1024")
        return _number(_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp):
        value = _interpret(node.operand, values)
        if isinstance(node.op, ast.Not):
            return not value
        value = _number(value)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Compare):
        left = _interpret(node.left, values)
        for op, expression in zip(node.ops, node.comparators):
            right = _interpret(expression, values)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        for expression in node.values:
            value = _interpret(expression, values)
            if isinstance(node.op, ast.And) and not value:
                return value
            if isinstance(node.op, ast.Or) and value:
                return value
        return value
    if isinstance(node, ast.IfExp):
        branch = node.body if _interpret(node.test, values) else node.orelse
        return _interpret(branch, values)
    if isinstance(node, ast.Call):
        args = [_number(_interpret(arg, values)) for arg in node.args]
        return _number(_CALLS[node.func.id][0](*args))
    raise ValueError("Unsupported expression syntax")  # Validated before evaluation.


def _compile_value(expression: str, arguments: tuple[str, ...]) -> Callable[..., Any]:
    """Validate syntax once, sharing the interpreter across scalar and label results."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a nonempty string")
    if len(expression) > _MAX_LENGTH:
        raise ValueError("Expression exceeds 4096 characters")
    if (not isinstance(arguments, tuple)
            or any(type(name) is not str or not name.isidentifier() or keyword.iskeyword(name)
                   for name in arguments)
            or len(set(arguments)) != len(arguments)
            or any(name in _CALLS or name == "pi" for name in arguments)):
        raise ValueError("Expression arguments must be distinct, nonreserved identifier names")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("Expression exceeds 256 syntax nodes")
    _validate(tree.body, set(arguments))

    def evaluate(*args: Any) -> Any:
        if len(args) != len(arguments):
            raise ValueError(f"Expression expects {len(arguments)} arguments; got {len(args)}")
        try:
            for value in args:
                _argument(value)
            return _interpret(tree.body, dict(zip(arguments, args)))
        except (ArithmeticError, TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"Could not evaluate symbol expression: {exc}") from exc

    return evaluate


def compile_expression(expression: str, arguments: tuple[str, ...], *,
                       preserve_numeric_type: bool = False) -> Callable[..., int | float | complex]:
    """Return a finite numeric symbol function, rejecting boolean results.

    Expressions have at most 4096 characters, 256 syntax nodes and 32 nested
    levels. Numeric intermediates have magnitude at most ``1e100`` and powers
    have exponent magnitude at most 1024. Labels may be strings, integers or
    integer tuples/lists. ``preserve_numeric_type`` retains integer/real results
    for typed fields such as fusion multiplicities without complex rounding.
    Invalid syntax and failed evaluations raise :class:`ValueError`.
    """
    evaluate = _compile_value(expression, arguments)

    def symbol(*args: Any) -> int | float | complex:
        result = _number(evaluate(*args))
        return result if preserve_numeric_type else complex(result)

    return symbol


def compile_action(expression: str, arguments: tuple[str, ...] = ("g", "a")) -> Callable[..., Any]:
    """Return a label-valued action using the same bounded expression grammar."""
    evaluate = _compile_value(expression, arguments)

    def action(*args: Any) -> str | int | tuple[int, ...]:
        result = evaluate(*args)
        if type(result) in (str, int):
            _argument(result)
            return result
        if type(result) in (tuple, list):
            _integer_sequence(result)
            return tuple(result)
        raise ValueError("Action must return a string, integer, or integer tuple anyon label")

    return action
