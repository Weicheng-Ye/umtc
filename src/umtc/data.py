"""Read tabulated or function-defined symbols through one callable interface.

Tables are complete on admissible channels. JSON functions use a restricted
mathematical expression grammar; native Python functions can also be supplied
programmatically. Multiplicity indices are zero based in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import itertools
import inspect
import json
import math
from numbers import Number
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeAlias

from .expressions import compile_action, compile_expression
from .gap import GAPError, GAPGroup, GAPZ2Homomorphism

Label: TypeAlias = str | int | tuple[int, ...]
Matrix: TypeAlias = tuple[tuple[complex, ...], ...]
Basis: TypeAlias = tuple[tuple[Label, int, int], ...]


class DataError(ValueError):
    """Malformed data, missing symbols, or an unknown label."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataError(f"{where} must be a list")
    return value


def _required(record: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in record:
        raise DataError(f"{where} is missing {key!r}")
    return record[key]


def _label(value: Any, where: str = "anyon") -> Label:
    if isinstance(value, str) or type(value) is int:
        return value
    if isinstance(value, (list, tuple)) and all(type(x) is int for x in value):
        return tuple(value)
    raise DataError(f"{where} must be a string, integer, or tuple of integers")


def _real(value: Any, where: str) -> float:
    if type(value) not in (int, float):
        raise DataError(f"{where} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise DataError(f"{where} must be finite") from exc
    if not math.isfinite(result):
        raise DataError(f"{where} must be finite")
    return result


def _complex(value: Any, where: str) -> complex:
    if isinstance(value, Mapping):
        if set(value) != {"re", "im"}:
            raise DataError(f"{where} must have exactly 're' and 'im' components")
        return complex(_real(value["re"], f"{where}.re"), _real(value["im"], f"{where}.im"))
    return complex(_real(value, where))


def _matrix(value: Any, rows: int, columns: int, where: str) -> Matrix:
    raw_rows = _list(value, where)
    if len(raw_rows) != rows:
        raise DataError(f"{where} must have shape {rows} x {columns}; got {len(raw_rows)} rows")
    result = []
    for i, raw_row in enumerate(raw_rows):
        row = _list(raw_row, f"{where}[{i}]")
        if len(row) != columns:
            raise DataError(f"{where} must have shape {rows} x {columns}; row {i} has {len(row)} columns")
        result.append(tuple(_complex(x, f"{where}[{i}][{j}]") for j, x in enumerate(row)))
    return tuple(result)


def _indices(*indices: int) -> None:
    if any(type(i) is not int or i < 0 for i in indices):
        raise DataError("Multiplicity indices must be nonnegative integers")


def _anyons(raw: Any) -> tuple[Label, ...]:
    if isinstance(raw, Mapping):
        if raw.get("type") != "tuples":
            raise DataError("anyons.type must be 'tuples'")
        if ("moduli" in raw) == ("labels" in raw):
            raise DataError("Tuple anyons must specify exactly one of 'moduli' and 'labels'")
        if "moduli" in raw:
            moduli = _list(raw["moduli"], "anyons.moduli")
            if any(type(n) is not int or n <= 0 for n in moduli):
                raise DataError("anyons.moduli must contain positive integers")
            values = list(itertools.product(*(range(n) for n in moduli)))
        else:
            values = _list(raw["labels"], "anyons.labels")
            if any(not isinstance(x, (list, tuple)) for x in values):
                raise DataError("anyons.labels must contain integer tuples")
    else:
        values = _list(raw, "anyons")
    labels = tuple(_label(value, f"anyons[{i}]") for i, value in enumerate(values))
    if not labels:
        raise DataError("anyons must not be empty")
    if len(set(labels)) != len(labels):
        raise DataError("anyons contains duplicate labels")
    tuple_lengths = {len(label) for label in labels if isinstance(label, tuple)}
    if len(tuple_lengths) > 1:
        raise DataError("Tuple anyon labels must have a common length")
    return labels


def _coverage(actual: Mapping[Any, Any], expected: set[Any], where: str) -> None:
    missing = expected.difference(actual)
    if missing:
        example = next(iter(missing))
        raise DataError(f"{where} is missing {len(missing)} required records, including {example!r}")


def _insert(table: dict[Any, Any], key: Any, value: Any, where: str) -> None:
    if key in table:
        raise DataError(f"{where} contains duplicate record {key!r}")
    table[key] = value


def _freeze(values: dict[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(values)


_F_ARGUMENTS = ("a", "b", "c", "d", "e", "f", "alpha", "beta", "mu", "nu")
_R_ARGUMENTS = ("a", "b", "c", "mu", "nu")
_N_ARGUMENTS = ("a", "b", "c")
_U_ARGUMENTS = ("g", "a", "b", "c", "mu", "nu")
_ETA_ARGUMENTS = ("a", "g", "h")


def _symbol_format(data: Mapping[str, Any]) -> str:
    """Infer a common representation, or verify an explicitly selected one."""
    formats = []
    for name in ("F_symbols", "R_symbols"):
        raw = _required(data, name, "UMTC data")
        if isinstance(raw, list):
            formats.append("tables")
        elif isinstance(raw, Mapping) or callable(raw):
            formats.append("functions")
        else:
            raise DataError(f"{name} must be a table list, expression object, or Python callable")
    if formats[0] != formats[1]:
        raise DataError("F_symbols and R_symbols must both use tables or both use functions")
    selected = data.get("symbol_format", formats[0])
    if selected not in ("tables", "functions"):
        raise DataError("symbol_format must be 'tables' or 'functions'")
    if selected != formats[0]:
        raise DataError(f"symbol_format={selected!r} does not match the F/R definitions")
    return selected


def _is_function(raw: Any, symbol_format: str, where: str) -> bool:
    """Function mode also accepts existing tables for individual extra fields."""
    if isinstance(raw, list):
        return False
    if symbol_format != "functions":
        raise DataError(f"{where} must be an explicit table in 'tables' mode")
    if not isinstance(raw, Mapping) and not callable(raw):
        raise DataError(f"{where} must be a table list, expression object, or Python callable")
    return True


def _symbol_function(raw: Any, arguments: tuple[str, ...], anyon_count: int,
                     multiplicities: bool, where: str, *,
                     result_kind: str = "complex") -> Callable[..., Any]:
    """Adapt a formula or native function once; evaluate only on demand."""
    if isinstance(raw, Mapping):
        if set(raw) != {"expression"}:
            raise DataError(f"{where} function must have exactly an 'expression' field")
        try:
            if result_kind == "label":
                function = compile_action(raw["expression"], arguments)
            else:
                function = compile_expression(raw["expression"], arguments,
                                              preserve_numeric_type=result_kind != "complex")
        except ValueError as exc:
            raise DataError(f"{where}: {exc}") from exc
        invoke = function
    elif callable(raw):
        try:
            signature = inspect.signature(raw)
        except (ValueError, TypeError) as exc:
            raise DataError(f"{where} must have an inspectable Python function signature") from exc

        def accepts(*args, **kwargs):
            try:
                signature.bind(*args, **kwargs)
                return True
            except TypeError:
                return False

        placeholders = (None,) * len(arguments)
        keyword_indices = dict.fromkeys(arguments[anyon_count:], 0)
        has_keyword_indices = any(
            name in keyword_indices and parameter.kind == inspect.Parameter.KEYWORD_ONLY
            for name, parameter in signature.parameters.items())
        accepts_keywords = accepts(*placeholders[:anyon_count], **keyword_indices)
        accepts_positional = accepts(*placeholders)
        # *args must not swallow explicit keyword-only indices. Conversely,
        # **kwargs must not swallow indices intended for positional parameters
        # with noncanonical names (e.g. row,column instead of mu,nu).
        if not has_keyword_indices and accepts_positional:
            invoke = raw
        elif accepts_keywords:
            def invoke(*values):
                return raw(*values[:anyon_count], **dict(zip(arguments[anyon_count:], values[anyon_count:])))
        elif not multiplicities and accepts(*placeholders[:anyon_count]):
            def invoke(*values):
                return raw(*values[:anyon_count])
        else:
            suffix = "; fusion multiplicities require the index arguments" if multiplicities else ""
            raise DataError(f"{where} must accept arguments {arguments}{suffix}")
    else:
        raise DataError(f"{where} must be an expression object or Python callable")

    def evaluate(*values):
        try:
            result = invoke(*values)
            if result_kind == "label":
                return _label(result, "action result")
            if isinstance(result, bool) or not isinstance(result, Number):
                raise ValueError("symbol function must return a numeric scalar")
            if result_kind == "multiplicity":
                if type(result) is int:
                    integer = result
                else:
                    numeric = complex(result)
                    if (not math.isfinite(numeric.real) or numeric.imag != 0
                            or not numeric.real.is_integer()):
                        raise ValueError("fusion multiplicity must be a nonnegative integer")
                    integer = int(numeric.real)
                    if result != integer:
                        raise ValueError("fusion multiplicity must be an exact integer")
                if integer < 0:
                    raise ValueError("fusion multiplicity must be a nonnegative integer")
                return integer
            result = complex(result)
            if not math.isfinite(result.real) or not math.isfinite(result.imag):
                raise ValueError("symbol function must return a finite number")
            if result_kind == "real":
                if result.imag != 0:
                    raise ValueError("quantum dimension must be real")
                return result.real
            return result
        except Exception as exc:
            # Include the actual channel, not just a formula-engine traceback.
            # Do not retry with another signature when user code raises TypeError.
            labels = dict(zip(arguments, values))
            raise DataError(f"{where} evaluation failed at {labels!r}: {exc}") from exc
    return evaluate


@dataclass(frozen=True)
class UMTC:
    """A numerical category with a common callable interface for all data.

    Construct with :meth:`from_dict`, :meth:`from_json`, or :meth:`from_functions`.
    Table data are wrapped as lookup functions. Formula and native functions
    use the same admissibility and indexing rules. Fusion is evaluated and
    cached while constructing the finite fusion graph; other functions are lazy.
    """

    name: str
    anyons: tuple[Label, ...]
    vacuum: Label
    _fusion_rules: Mapping[tuple[Label, Label, Label], int] = field(repr=False)
    _N: Callable[..., int] = field(repr=False)
    _dimension: Callable[[Label], float] = field(repr=False)
    _spin: Callable[[Label], complex] = field(repr=False)
    _F: Callable[..., complex] = field(repr=False)
    _R: Callable[..., complex] = field(repr=False)
    _left_bases: Mapping[tuple[Label, Label, Label, Label], Basis] = field(repr=False)
    _right_bases: Mapping[tuple[Label, Label, Label, Label], Basis] = field(repr=False)
    symbol_format: str
    symmetry: Symmetry | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UMTC:
        """Parse a schema-version-1 dictionary without checking its equations."""
        data = _mapping(data, "UMTC data")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise DataError("schema_version must be the integer 1")
        symbol_format = _symbol_format(data)
        anyons = _anyons(_required(data, "anyons", "UMTC data"))
        anyon_set = set(anyons)

        def known(value: Any, where: str) -> Label:
            label = _label(value, where)
            if label not in anyon_set:
                raise DataError(f"{where} refers to unknown anyon {label!r}")
            return label

        def records(name: str):
            for i, raw_record in enumerate(_list(_required(data, name, "UMTC data"), name)):
                where = f"{name}[{i}]"
                yield _mapping(raw_record, where), where

        def key(record: Mapping[str, Any], names: str, where: str) -> tuple[Label, ...]:
            return tuple(known(_required(record, k, where), f"{where}.{k}") for k in names)

        vacuum = known(_required(data, "vacuum", "UMTC data"), "vacuum")
        name = data.get("name", "")
        if not isinstance(name, str):
            raise DataError("name must be a string")
        fusion = {}
        raw_fusion = _required(data, "fusion_rules", "UMTC data")
        if _is_function(raw_fusion, symbol_format, "fusion_rules"):
            n_function = cache(_symbol_function(raw_fusion, _N_ARGUMENTS, 3, False,
                                                "fusion_rules", result_kind="multiplicity"))
            for channel in itertools.product(anyons, repeat=3):
                multiplicity = n_function(*channel)
                if multiplicity:
                    fusion[channel] = multiplicity
        else:
            for record, where in records("fusion_rules"):
                channel = key(record, "abc", where)
                multiplicity = _required(record, "multiplicity", where)
                if type(multiplicity) is not int or multiplicity <= 0:
                    raise DataError(f"{where}.multiplicity must be a positive integer")
                _insert(fusion, channel, multiplicity, "fusion_rules")

            def n_function(a, b, c):
                return fusion.get((a, b, c), 0)

        raw_dimensions = _required(data, "quantum_dimensions", "UMTC data")
        if _is_function(raw_dimensions, symbol_format, "quantum_dimensions"):
            dimension_function = _symbol_function(raw_dimensions, ("a",), 1, False,
                                                  "quantum_dimensions", result_kind="real")
        else:
            dimensions = {}
            for record, where in records("quantum_dimensions"):
                a = known(_required(record, "anyon", where), f"{where}.anyon")
                value = _real(_required(record, "value", where), f"{where}.value")
                _insert(dimensions, a, value, "quantum_dimensions")
            _coverage(dimensions, anyon_set, "quantum_dimensions")

            def dimension_function(a):
                return dimensions[a]

        raw_spins = _required(data, "topological_spins", "UMTC data")
        if _is_function(raw_spins, symbol_format, "topological_spins"):
            spin_function = _symbol_function(raw_spins, ("a",), 1, False, "topological_spins")
        else:
            spins = {}
            for record, where in records("topological_spins"):
                a = known(_required(record, "anyon", where), f"{where}.anyon")
                value = _complex(_required(record, "value", where), f"{where}.value")
                _insert(spins, a, value, "topological_spins")
            _coverage(spins, anyon_set, "topological_spins")

            def spin_function(a):
                return spins[a]

        left_bases, right_bases = {}, {}
        for a, b, c, d in itertools.product(anyons, repeat=4):
            channel = (a, b, c, d)
            left = tuple((e, alpha, beta) for e in anyons
                         for alpha in range(fusion.get((a, b, e), 0))
                         for beta in range(fusion.get((e, c, d), 0)))
            right = tuple((f, mu, nu) for f in anyons
                          for mu in range(fusion.get((b, c, f), 0))
                          for nu in range(fusion.get((a, f, d), 0)))
            if left or right:
                left_bases[channel], right_bases[channel] = left, right
        if symbol_format == "tables":
            f_symbols = {}
            for record, where in records("F_symbols"):
                channel = key(record, "abcd", where)
                if channel not in left_bases:
                    raise DataError(f"{where} specifies an inadmissible F channel {channel!r}")
                value = _matrix(_required(record, "matrix", where), len(left_bases[channel]),
                                len(right_bases[channel]), f"{where}.matrix")
                _insert(f_symbols, channel, value, "F_symbols")
            _coverage(f_symbols, set(left_bases), "F_symbols")
            r_symbols = {}
            for record, where in records("R_symbols"):
                channel = key(record, "abc", where)
                a, b, c = channel
                if channel not in fusion:
                    raise DataError(f"{where} specifies an inadmissible R channel {channel!r}")
                value = _matrix(_required(record, "matrix", where), fusion.get((b, a, c), 0),
                                fusion[channel], f"{where}.matrix")
                _insert(r_symbols, channel, value, "R_symbols")
            _coverage(r_symbols, set(fusion), "R_symbols")

            def f_function(a, b, c, d, e, f, alpha, beta, mu, nu):
                channel = a, b, c, d
                row = left_bases[channel].index((e, alpha, beta))
                column = right_bases[channel].index((f, mu, nu))
                return f_symbols[channel][row][column]

            def r_function(a, b, c, mu, nu):
                return r_symbols[(a, b, c)][mu][nu]
        else:
            multiplicities = any(n > 1 for n in fusion.values())
            f_function = _symbol_function(data["F_symbols"], _F_ARGUMENTS, 6,
                                          multiplicities, "F_symbols")
            r_function = _symbol_function(data["R_symbols"], _R_ARGUMENTS, 3,
                                          multiplicities, "R_symbols")
        category = cls(name, anyons, vacuum, _freeze(fusion), n_function, dimension_function,
                       spin_function, f_function, r_function,
                       _freeze(left_bases), _freeze(right_bases), symbol_format)
        if data.get("symmetry") is not None:
            object.__setattr__(category, "symmetry", Symmetry._from_dict(data["symmetry"], category))
        return category

    @classmethod
    def from_functions(cls, data: Mapping[str, Any], *,
                       F: Callable[..., complex] | None = None,
                       R: Callable[..., complex] | None = None,
                       N: Callable[..., int] | None = None,
                       quantum_dimension: Callable[[Label], float] | None = None,
                       spin: Callable[[Label], complex] | None = None,
                       U: Callable[..., complex] | None = None,
                       eta: Callable[..., complex] | None = None,
                       action: Callable[[str, Label], Label] | None = None) -> UMTC:
        """Override data with deterministic native Python symbol functions.

        Omitted/None arguments preserve existing definitions. Fields replaced by
        callbacks may be omitted from ``data``. F/R must both be function
        definitions in the result, except that an action-only override preserves
        the existing F/R mode. U/eta/action require symmetry group data.
        N is evaluated once per anyon triple while constructing the fusion
        spaces; dimensions, spins and F/R/U/eta are evaluated only when used.
        Multiplicity-free F/R/U callbacks may omit multiplicity arguments.
        The input dictionary is not mutated.
        """
        _mapping(data, "UMTC data")
        result = dict(data)
        if action is None or any(callback is not None for callback in
                                  (F, R, N, quantum_dimension, spin, U, eta)):
            result["symbol_format"] = "functions"
        for field_name, callback in (("F_symbols", F), ("R_symbols", R), ("fusion_rules", N),
                                     ("quantum_dimensions", quantum_dimension), ("topological_spins", spin)):
            if callback is not None:
                if not callable(callback):
                    raise DataError(f"{field_name} override must be a Python callable")
                result[field_name] = callback
        if U is not None or eta is not None or action is not None:
            if not isinstance(data.get("symmetry"), Mapping):
                raise DataError("U/eta/action overrides require explicit symmetry group data")
            symmetry = dict(data["symmetry"])
            for field_name, callback in (("U_symbols", U), ("eta_symbols", eta), ("action", action)):
                if callback is not None:
                    if not callable(callback):
                        raise DataError(f"{field_name} override must be a Python callable")
                    symmetry[field_name] = callback
            result["symmetry"] = symmetry
        return cls.from_dict(result)

    @classmethod
    def from_json(cls, path: str | Path) -> UMTC:
        """Load UTF-8 JSON from a filesystem path, rejecting duplicate keys."""
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise DataError(f"JSON object contains duplicate key {key!r}")
                result[key] = value
            return result

        try:
            with Path(path).open(encoding="utf-8") as handle:
                data = json.load(handle, object_pairs_hook=unique_object)
        except json.JSONDecodeError as exc:
            raise DataError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    def _known(self, a: Label) -> Label:
        a = _label(a)
        if a not in self.anyons:
            raise DataError(f"Unknown anyon {a!r}")
        return a

    def N(self, a: Label, b: Label, c: Label) -> int:
        """Fusion multiplicity, zero for an absent fusion channel."""
        return self._N(*(self._known(x) for x in (a, b, c)))

    def fusion(self, a: Label, b: Label) -> dict[Label, int]:
        a, b = self._known(a), self._known(b)
        return {c: n for c in self.anyons if (n := self.N(a, b, c))}

    def F_left_basis(self, a: Label, b: Label, c: Label, d: Label) -> Basis:
        """Ordered ``(e, alpha, beta)`` basis of ``(a b) c -> d``."""
        return self._left_bases.get(tuple(self._known(x) for x in (a, b, c, d)), ())

    def F_right_basis(self, a: Label, b: Label, c: Label, d: Label) -> Basis:
        """Ordered ``(f, mu, nu)`` basis of ``a (b c) -> d``."""
        return self._right_bases.get(tuple(self._known(x) for x in (a, b, c, d)), ())

    def F_matrix(self, a: Label, b: Label, c: Label, d: Label) -> Matrix:
        """Build the matrix from the same scalar F function used by checkers."""
        a, b, c, d = (self._known(x) for x in (a, b, c, d))
        left, right = self.F_left_basis(a, b, c, d), self.F_right_basis(a, b, c, d)
        return tuple(tuple(self.F(a, b, c, d, e, f, alpha, beta, mu, nu)
                           for f, mu, nu in right) for e, alpha, beta in left)

    def F(self, a: Label, b: Label, c: Label, d: Label, e: Label, f: Label,
          alpha: int = 0, beta: int = 0, mu: int = 0, nu: int = 0) -> complex:
        """F entry with left row ``(e, alpha, beta)`` and right column ``(f, mu, nu)``.

        Known but inadmissible channels or multiplicity indices have value zero.
        """
        a, b, c, d, e, f = (self._known(x) for x in (a, b, c, d, e, f))
        _indices(alpha, beta, mu, nu)
        left, right = self.F_left_basis(a, b, c, d), self.F_right_basis(a, b, c, d)
        if (e, alpha, beta) not in left or (f, mu, nu) not in right:
            return 0j
        return self._F(a, b, c, d, e, f, alpha, beta, mu, nu)

    def R_matrix(self, a: Label, b: Label, c: Label) -> Matrix:
        """Build the matrix from the same scalar R function used by checkers."""
        a, b, c = (self._known(x) for x in (a, b, c))
        n_ab, n_ba = self.N(a, b, c), self.N(b, a, c)
        if not n_ab:
            return ()
        return tuple(tuple(self.R(a, b, c, mu, nu) for nu in range(n_ab)) for mu in range(n_ba))

    def R(self, a: Label, b: Label, c: Label, mu: int = 0, nu: int = 0) -> complex:
        """R entry with ``b a -> c`` row ``mu`` and ``a b -> c`` column ``nu``."""
        a, b, c = (self._known(x) for x in (a, b, c))
        _indices(mu, nu)
        if mu >= self.N(b, a, c) or nu >= self.N(a, b, c):
            return 0j
        return self._R(a, b, c, mu, nu)

    @property
    def F_symbols(self) -> Callable[..., complex]:
        """Callable alias of :meth:`F`, for either input representation."""
        return self.F

    @property
    def R_symbols(self) -> Callable[..., complex]:
        """Callable alias of :meth:`R`, for either input representation."""
        return self.R

    def quantum_dimension(self, a: Label) -> float:
        return self._dimension(self._known(a))

    def spin(self, a: Label) -> complex:
        return self._spin(self._known(a))

    @property
    def fusion_rules(self) -> Callable[..., int]:
        """Callable alias of :meth:`N` in either representation."""
        return self.N

    @property
    def quantum_dimensions(self) -> Callable[[Label], float]:
        """Callable alias of :meth:`quantum_dimension`."""
        return self.quantum_dimension

    @property
    def topological_spins(self) -> Callable[[Label], complex]:
        """Callable alias of :meth:`spin`."""
        return self.spin

    @property
    def total_quantum_dimension(self) -> float:
        return math.hypot(*(self.quantum_dimension(a) for a in self.anyons))

    def check_pentagon(self, **kwargs: Any) -> bool:
        from .checks import check_pentagon
        return check_pentagon(self, **kwargs)

    def check_hexagon(self, **kwargs: Any) -> bool:
        from .checks import check_hexagon
        return check_hexagon(self, **kwargs)

    def check_symmetry(self, **kwargs: Any) -> bool:
        from .checks import check_symmetry
        return check_symmetry(self, **kwargs)

    def check_all(self, **kwargs: Any) -> bool:
        from .checks import check_all
        return check_all(self, **kwargs)

    def coherence_report(self, **kwargs: Any):
        from .checks import coherence_report
        return coherence_report(self, **kwargs)

    def check_fusion(self, **kwargs: Any) -> bool:
        from .checks import check_fusion
        return check_fusion(self, **kwargs)

    def check_unitarity(self, **kwargs: Any) -> bool:
        from .checks import check_unitarity
        return check_unitarity(self, **kwargs)

    def check_ribbon(self, **kwargs: Any) -> bool:
        from .checks import check_ribbon
        return check_ribbon(self, **kwargs)

    def check_modularity(self, **kwargs: Any) -> bool:
        from .checks import check_modularity
        return check_modularity(self, **kwargs)


@dataclass(frozen=True)
class Symmetry:
    """Finite intrinsic symmetry with GAP group, parity homomorphism and action.

    ``G`` is the GAP-backed finite group and ``rho(g)`` is 0 for unitary or
    1 for antiunitary elements. Legacy table input has ``G=None``. Both action
    representations expose the same callable ``action(g, a)``/``act(g, a)``.

    U labels are destination labels in the convention of arXiv:2210.02444:
    ``U(g, a, b, c)`` describes the transformed fusion vertex labelled by
    ``a, b, c``. No multiplicity-space identities are implicit.
    """

    elements: tuple[str, ...]
    identity: str
    multiplication_table: tuple[tuple[str, ...], ...]
    antiunitary: frozenset[str]
    G: GAPGroup | None
    rho: Callable[[str], int] = field(repr=False)
    _action: Callable[[str, Label], Label] = field(repr=False)
    _U: Callable[..., complex] = field(repr=False)
    _eta: Callable[..., complex] = field(repr=False)
    _category: UMTC = field(repr=False, compare=False)

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any], category: UMTC) -> Symmetry:
        data = _mapping(data, "symmetry")
        gap_group = None
        if "group" in data:
            conflicts = {"elements", "identity", "multiplication_table", "antiunitary"}.intersection(data)
            if conflicts:
                raise DataError(f"GAP symmetry cannot also specify legacy fields: {sorted(conflicts)!r}")
            raw_group = data["group"]
            try:
                if isinstance(raw_group, GAPGroup):
                    gap_group = raw_group
                else:
                    raw_group = _mapping(raw_group, "symmetry.group")
                    if set(raw_group) - {"gap", "generator_names"}:
                        raise DataError("symmetry.group accepts only 'gap' and 'generator_names'")
                    gap_group = GAPGroup.from_gap(_required(raw_group, "gap", "symmetry.group"),
                                                  raw_group.get("generator_names"))
                raw_rho = _required(data, "rho", "symmetry")
                if isinstance(raw_rho, GAPZ2Homomorphism):
                    if raw_rho.source is not gap_group:
                        raise DataError("symmetry.rho must have symmetry.group as its source")
                    rho = raw_rho
                else:
                    raw_rho = _mapping(raw_rho, "symmetry.rho")
                    if set(raw_rho) != {"generator_images"}:
                        raise DataError("symmetry.rho must have exactly 'generator_images'")
                    rho = gap_group.homomorphism_to_z2(raw_rho["generator_images"])
            except GAPError as exc:
                raise DataError(f"symmetry: {exc}") from exc
            elements = gap_group.elements
        else:
            if "rho" in data:
                raise DataError("symmetry.rho requires a GAP symmetry.group")
            elements = tuple(_list(_required(data, "elements", "symmetry"), "symmetry.elements"))
            if not elements or any(not isinstance(g, str) for g in elements):
                raise DataError("symmetry.elements must be a nonempty list of strings")
            if len(set(elements)) != len(elements):
                raise DataError("symmetry.elements contains duplicate labels")

        def group(value: Any, where: str) -> str:
            if not isinstance(value, str) or value not in elements:
                raise DataError(f"{where} refers to unknown symmetry element {value!r}")
            return value

        def records(name: str):
            for i, raw in enumerate(_list(_required(data, name, "symmetry"), f"symmetry.{name}")):
                where = f"symmetry.{name}[{i}]"
                yield _mapping(raw, where), where

        if gap_group is not None:
            identity, table = gap_group.identity, gap_group.multiplication_table
            anti = frozenset(g for g in elements if rho(g) == 1)
        else:
            identity = group(_required(data, "identity", "symmetry"), "symmetry.identity")
            raw_table = _list(_required(data, "multiplication_table", "symmetry"), "symmetry.multiplication_table")
            if len(raw_table) != len(elements):
                raise DataError("symmetry.multiplication_table must be square with one row per element")
            table = []
            for i, raw in enumerate(raw_table):
                row = _list(raw, f"symmetry.multiplication_table[{i}]")
                if len(row) != len(elements):
                    raise DataError("symmetry.multiplication_table must be square with one column per element")
                table.append(tuple(group(g, "symmetry.multiplication_table") for g in row))
            raw_anti = _list(_required(data, "antiunitary", "symmetry"), "symmetry.antiunitary")
            anti = tuple(group(g, "symmetry.antiunitary") for g in raw_anti)
            if len(set(anti)) != len(anti):
                raise DataError("symmetry.antiunitary contains duplicate labels")
            anti = frozenset(anti)

            def rho(g):
                return int(group(g, "symmetry.rho") in anti)

        raw_action = _required(data, "action", "symmetry")
        if isinstance(raw_action, list):
            action_table = {}
            for record, where in records("action"):
                g = group(_required(record, "g", where), f"{where}.g")
                raw_images = _list(_required(record, "images", where), f"{where}.images")
                if len(raw_images) != len(category.anyons):
                    raise DataError(f"{where}.images must have one image per anyon")
                _insert(action_table, g, tuple(category._known(a) for a in raw_images), "symmetry.action")
            _coverage(action_table, set(elements), "symmetry.action")

            def action_function(g, a):
                return action_table[g][category.anyons.index(a)]
        else:
            # The group action can be a function independently of the F/R mode.
            provider = _symbol_function(raw_action, ("g", "a"), 2, False,
                                        "symmetry.action", result_kind="label")

            def action_function(g, a):
                result = provider(g, a)
                try:
                    return category._known(result)
                except DataError as exc:
                    raise DataError(f"symmetry.action evaluation failed at g={g!r}, a={a!r}: {exc}") from exc
        raw_u = _required(data, "U_symbols", "symmetry")
        if _is_function(raw_u, category.symbol_format, "symmetry.U_symbols"):
            multiplicities = any(n > 1 for n in category._fusion_rules.values())
            u_function = _symbol_function(raw_u, _U_ARGUMENTS, 4, multiplicities, "symmetry.U_symbols")
        else:
            u_symbols = {}
            for record, where in records("U_symbols"):
                g = group(_required(record, "g", where), f"{where}.g")
                a, b, c = (category._known(_required(record, k, where)) for k in "abc")
                n = category.N(a, b, c)
                if not n:
                    raise DataError(f"{where} specifies an inadmissible U channel {(g, a, b, c)!r}")
                value = _matrix(_required(record, "matrix", where), n, n, f"{where}.matrix")
                _insert(u_symbols, (g, a, b, c), value, "symmetry.U_symbols")
            _coverage(u_symbols, {(g, a, b, c) for g in elements for a, b, c in category._fusion_rules}, "symmetry.U_symbols")

            def u_function(g, a, b, c, mu, nu):
                return u_symbols[(g, a, b, c)][mu][nu]

        raw_eta = _required(data, "eta_symbols", "symmetry")
        if _is_function(raw_eta, category.symbol_format, "symmetry.eta_symbols"):
            eta_function = _symbol_function(raw_eta, _ETA_ARGUMENTS, 3, False, "symmetry.eta_symbols")
        else:
            eta_symbols = {}
            for record, where in records("eta_symbols"):
                a = category._known(_required(record, "anyon", where))
                g = group(_required(record, "g", where), f"{where}.g")
                h = group(_required(record, "h", where), f"{where}.h")
                value = _complex(_required(record, "value", where), f"{where}.value")
                _insert(eta_symbols, (a, g, h), value, "symmetry.eta_symbols")
            _coverage(eta_symbols, set(itertools.product(category.anyons, elements, elements)), "symmetry.eta_symbols")

            def eta_function(a, g, h):
                return eta_symbols[(a, g, h)]
        return cls(elements, identity, tuple(table), anti, gap_group, rho,
                   action_function, u_function, eta_function, category)

    def _known(self, g: str) -> str:
        if not isinstance(g, str) or g not in self.elements:
            raise DataError(f"Unknown symmetry element {g!r}")
        return g

    def is_antiunitary(self, g: str) -> bool:
        return self.rho(self._known(g)) == 1

    def mul(self, g: str, h: str) -> str:
        g, h = self._known(g), self._known(h)
        if self.G is not None:
            return self.G.mul(g, h)
        return self.multiplication_table[self.elements.index(g)][self.elements.index(h)]

    def inverse(self, g: str) -> str:
        g = self._known(g)
        if self.G is not None:
            return self.G.inverse(g)
        inverses = [h for h in self.elements if self.mul(g, h) == self.identity and self.mul(h, g) == self.identity]
        if len(inverses) != 1:
            raise DataError(f"Symmetry element {g!r} has no unique two-sided inverse")
        return inverses[0]

    def act(self, g: str, a: Label) -> Label:
        g, a = self._known(g), self._category._known(a)
        return self._action(g, a)

    @property
    def action(self) -> Callable[[str, Label], Label]:
        """Callable anyon action, whether supplied as a function or a table."""
        return self.act

    def U_matrix(self, g: str, a: Label, b: Label, c: Label) -> Matrix:
        g = self._known(g)
        a, b, c = (self._category._known(x) for x in (a, b, c))
        n = self._category.N(a, b, c)
        return tuple(tuple(self.U(g, a, b, c, mu, nu) for nu in range(n)) for mu in range(n))

    def U(self, g: str, a: Label, b: Label, c: Label, mu: int = 0, nu: int = 0) -> complex:
        g = self._known(g)
        a, b, c = (self._category._known(x) for x in (a, b, c))
        _indices(mu, nu)
        n = self._category.N(a, b, c)
        if mu >= n or nu >= n:
            return 0j
        return self._U(g, a, b, c, mu, nu)

    def eta(self, a: Label, g: str, h: str) -> complex:
        return self._eta(self._category._known(a), self._known(g), self._known(h))

    @property
    def U_symbols(self) -> Callable[..., complex]:
        """Callable alias of :meth:`U` in either representation."""
        return self.U

    @property
    def eta_symbols(self) -> Callable[..., complex]:
        """Callable alias of :meth:`eta` in either representation."""
        return self.eta


def load_json(path: str | Path) -> UMTC:
    """Load a category from a JSON file."""
    return UMTC.from_json(path)
