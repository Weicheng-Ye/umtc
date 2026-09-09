"""Finite GAP groups and their homomorphisms to the additive group Z2.

GAP is an optional external executable; the rest of the package remains pure
Python. Group expressions use a deliberately small construction grammar:
``AbelianGroup``, ``CyclicGroup``, ``DihedralGroup``, ``SymmetricGroup``,
``AlternatingGroup``, ``SmallGroup``, ``Group``, and ``DirectProduct``, with
integer/list arguments, permutation cycles, and ``IsPermGroup``/``IsPcGroup``.
Arbitrary GAP programs, assignments, strings, and other calls are not accepted.
GAP has a 256 MiB workspace cap; permutation points are at most 1,000,000.

Elements have stable names obtained by breadth-first enumeration of positive
generator words. ``1`` denotes the identity; products use ``*``. The input
generator order is ``GeneratorsOfGroup(G)`` in GAP, including redundant ones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import re
import subprocess
from types import MappingProxyType
from typing import Mapping, Sequence


class GAPError(ValueError):
    """An invalid GAP construction, homomorphism, or backend response."""


class GAPUnavailableError(GAPError):
    """The external GAP executable could not be launched."""


_CONSTRUCTORS = frozenset({
    "AbelianGroup", "CyclicGroup", "DihedralGroup", "SymmetricGroup",
    "AlternatingGroup", "SmallGroup", "Group", "DirectProduct",
})
_CATEGORIES = frozenset({"IsPermGroup", "IsPcGroup"})
_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|-?[0-9]+|[\[\](),])")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_BEGIN = "__UMTC_GAP_BEGIN__"
_END = "__UMTC_GAP_END__"


class _ConstructionParser:
    """Validate and normalize a small data-only subset of GAP syntax."""

    def __init__(self, expression: str):
        if not isinstance(expression, str) or not expression.strip():
            raise GAPError("GAP group expression must be a nonempty string")
        if len(expression) > 16384:
            raise GAPError("GAP group expression is too long")
        self.tokens: list[str] = []
        position = 0
        while position < len(expression):
            match = _TOKEN.match(expression, position)
            if match is None:
                if expression[position:].isspace():
                    break
                raise GAPError("Unsupported syntax in GAP group expression")
            self.tokens.append(match.group(1))
            position = match.end()
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise GAPError("Malformed GAP group construction")
        self.index += 1
        return token

    def parse(self) -> str:
        if self.peek() not in _CONSTRUCTORS:
            raise GAPError("GAP group expression must start with a supported group constructor")
        result = self.value(0)
        if self.peek() is not None:
            raise GAPError("Unexpected trailing syntax in GAP group expression")
        return result

    def sequence(self, end: str, depth: int) -> str:
        values = []
        if self.peek() != end:
            values.append(self.value(depth + 1))
            while self.peek() == ",":
                self.take(",")
                values.append(self.value(depth + 1))
        self.take(end)
        return ",".join(values)

    def value(self, depth: int) -> str:
        if depth > 32:
            raise GAPError("GAP group expression nesting is too deep")
        token = self.take()
        if re.fullmatch(r"-?[0-9]+", token):
            # A small bound limits pathological integer parsing before GAP runs.
            if len(token) > 12:
                raise GAPError("Integer in GAP group expression is too large")
            return str(int(token))
        if token == "[":
            return "[" + self.sequence("]", depth) + "]"
        if token == "(":
            cycles = []
            while True:
                points = []
                if self.peek() != ")":
                    points.append(self.take())
                    while self.peek() == ",":
                        self.take(",")
                        points.append(self.take())
                self.take(")")
                if any(not re.fullmatch(r"[0-9]+", p) or len(p) > 7 or not 1 <= int(p) <= 1000000 for p in points):
                    raise GAPError("Permutation cycles require integer points between 1 and 1000000")
                points = [str(int(p)) for p in points]
                if len(set(points)) != len(points):
                    raise GAPError("Permutation cycle points must be distinct")
                cycles.append("(" + ",".join(points) + ")")
                if self.peek() != "(":
                    break
                self.take("(")
            return "".join(cycles)
        if token in _CATEGORIES:
            return token
        if token in _CONSTRUCTORS:
            self.take("(")
            return token + "(" + self.sequence(")", depth) + ")"
        raise GAPError(f"Unsupported name {token!r} in GAP group expression")


def _run_gap(script: str, executable: str, timeout: float) -> object:
    try:
        result = subprocess.run(
            [executable, "-q", "-b", "-r", "-A", "--quitonbreak", "-K", "256m", "-x", "1000000"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GAPError(f"GAP exceeded the {timeout:g}-second timeout") from exc
    except OSError as exc:
        raise GAPUnavailableError(
            f"Cannot launch GAP executable {executable!r}; install GAP or supply executable=..."
        ) from exc
    if result.returncode != 0 or re.search(r"(?m)^(?:Error,|Syntax error:)", result.stderr):
        detail = (result.stderr.strip() or result.stdout.strip())[-1500:]
        raise GAPError(f"GAP failed (exit {result.returncode}): {detail}")
    if result.stdout.count(_BEGIN) != 1 or result.stdout.count(_END) != 1:
        raise GAPError("GAP returned an invalid response (missing protocol markers)")
    payload = result.stdout.split(_BEGIN, 1)[1].split(_END, 1)[0]
    try:
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise GAPError("GAP returned an invalid response (malformed numeric data)") from exc


def _group_setup(expression: str, max_order: int) -> str:
    return f"""
umtcG := {expression};;
if not IsGroup(umtcG) then Error("Construction did not return a group"); fi;
if not IsFinite(umtcG) then Error("UMTC symmetry group must be finite"); fi;
umtcSize := Size(umtcG);;
if umtcSize < 1 or umtcSize > {max_order} then
  Error("UMTC symmetry group exceeds max_order or has invalid order");
fi;
umtcElements := Elements(umtcG);;
umtcGenerators := GeneratorsOfGroup(umtcG);;
umtcGeneratorPositions := List(umtcGenerators, x -> Position(umtcElements,x));;
umtcIdentity := Position(umtcElements,One(umtcG));;
"""


def _group_export() -> str:
    return r"""
Print("[",umtcSize,",",umtcIdentity,",",umtcGeneratorPositions,",[\n");
for umtcI in [1..umtcSize] do
  if umtcI > 1 then Print(",\n"); fi;
  Print(List(umtcElements,x -> Position(umtcElements,umtcElements[umtcI]*x)));
od;
Print("],",List(umtcElements,x -> Position(umtcElements,x^-1)),"]");
"""


def _validate_export(raw: object, max_order: int) -> tuple:
    if not isinstance(raw, list) or len(raw) != 5:
        raise GAPError("GAP returned an invalid group response")
    size, identity, generators, table, inverses = raw
    if type(size) is not int or not 1 <= size <= max_order:
        raise GAPError("GAP returned an invalid group order")

    def position(value: object) -> bool:
        return type(value) is int and 1 <= value <= size

    if not position(identity) or not isinstance(generators, list) or not all(map(position, generators)):
        raise GAPError("GAP returned invalid generator or identity positions")
    if not isinstance(table, list) or len(table) != size or any(
        not isinstance(row, list) or len(row) != size or not all(map(position, row)) for row in table
    ):
        raise GAPError("GAP returned an invalid multiplication table")
    if not isinstance(inverses, list) or len(inverses) != size or not all(map(position, inverses)):
        raise GAPError("GAP returned invalid inverses")
    table_tuple = tuple(tuple(x - 1 for x in row) for row in table)
    identity -= 1
    inverses_tuple = tuple(x - 1 for x in inverses)
    expected = set(range(size))
    if any(set(row) != expected for row in table_tuple) or any(
        {table_tuple[i][j] for i in range(size)} != expected for j in range(size)
    ):
        raise GAPError("GAP returned a noninvertible multiplication table")
    if any(
        table_tuple[identity][i] != i or table_tuple[i][identity] != i
        or table_tuple[i][inverses_tuple[i]] != identity
        or table_tuple[inverses_tuple[i]][i] != identity
        for i in range(size)
    ):
        raise GAPError("GAP returned inconsistent identity or inverses")
    return size, identity, tuple(x - 1 for x in generators), table_tuple, inverses_tuple


@dataclass(frozen=True, init=False, eq=False)
class GAPGroup:
    """An immutable finite GAP group with exact cached multiplication.

    Construct with :meth:`from_gap`. ``generators`` contains the canonical label
    of each GAP generator, in the order to use for rho's ``generator_images``.
    Identity or repeated generators can have the same canonical label. Distinct
    wrapper objects have distinct identity, including as homomorphism sources.
    """

    expression: str
    generator_names: tuple[str, ...]
    elements: tuple[str, ...]
    generators: tuple[str, ...]
    multiplication_table: tuple[tuple[str, ...], ...]
    identity: str = "1"
    _inverses: tuple[str, ...] = field(default=(), repr=False)
    _positions: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False, hash=False)
    _gap_export: tuple = field(default=(), repr=False)
    _order: tuple[int, ...] = field(default=(), repr=False)
    _executable: str = field(default="gap", repr=False)
    _timeout: float = field(default=30, repr=False)
    _max_order: int = field(default=256, repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise GAPError("Construct a GAPGroup with GAPGroup.from_gap(...)")

    @classmethod
    def from_gap(
        cls, expression: str, generator_names: Sequence[str] | None = None, *,
        executable: str = "gap", timeout: float = 30, max_order: int = 256,
    ) -> GAPGroup:
        """Ask GAP to construct and enumerate a finite group.

        ``generator_names`` names ``GeneratorsOfGroup(G)``, not an arbitrary
        generating subset. With no names, use ``g1``, ``g2``, ... . GAP must be
        installed only when loading a GAP-backed symmetry. Enumeration is
        limited to ``max_order`` elements and each GAP process to ``timeout``.
        """
        expression = _ConstructionParser(expression).parse()
        if not isinstance(executable, str) or not executable:
            raise GAPError("GAP executable must be a nonempty path or command name")
        try:
            valid_timeout = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise GAPError("GAP timeout must be a positive finite number")
        if type(max_order) is not int or max_order < 1:
            raise GAPError("GAP max_order must be a positive integer")
        if generator_names is None:
            names = None
        else:
            if isinstance(generator_names, (str, bytes)) or not isinstance(generator_names, Sequence):
                raise GAPError("GAP generator_names must be a sequence of distinct names")
            names = tuple(generator_names)
            if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in names):
                raise GAPError("GAP generator names must be identifiers; '1' and products are reserved")
            if len(set(names)) != len(names):
                raise GAPError("GAP generator names must be distinct")
        return _construct_group(expression, names, executable, float(timeout), max_order)

    def _position(self, element: str) -> int:
        try:
            return self._positions[element]
        except (KeyError, TypeError) as exc:
            raise GAPError(f"Unknown GAP group element {element!r}") from exc

    def mul(self, g: str, h: str) -> str:
        """Return the canonical label of the product g*h."""
        return self.multiplication_table[self._position(g)][self._position(h)]

    def inverse(self, g: str) -> str:
        return self._inverses[self._position(g)]

    def pow(self, g: str, n: int) -> str:
        self._position(g)
        if type(n) is not int:
            raise GAPError("Group powers must be integers")
        if n < 0:
            g, n = self.inverse(g), -n
        result = self.identity
        while n:
            if n % 2:
                result = self.mul(result, g)
            g = self.mul(g, g)
            n //= 2
        return result

    def homomorphism_to_z2(self, generator_images: Sequence[int]) -> GAPZ2Homomorphism:
        """Construct rho:G->Z2 with GAP's checked GroupHomomorphismByImages.

        Images are integer bits in the order of ``generators``. The map need
        not be surjective: all-zero images define unitary symmetry groups.
        """
        if isinstance(generator_images, (str, bytes)) or not isinstance(generator_images, Sequence):
            raise GAPError("rho generator_images must be a sequence of integer bits")
        bits = tuple(generator_images)
        if len(bits) != len(self.generators):
            raise GAPError(f"rho requires {len(self.generators)} generator images")
        if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
            raise GAPError("rho generator images must be integer 0 or 1")
        return _construct_homomorphism(self, bits)


@dataclass(frozen=True)
class GAPZ2Homomorphism:
    """A validated homomorphism to additive Z2; one means antiunitary.

    ``G.homomorphism_to_z2`` uses GAP's checked construction. Direct native
    construction also verifies every image and the complete homomorphism law.
    """

    source: GAPGroup
    generator_images: tuple[int, ...]
    images: tuple[int, ...]

    def __post_init__(self) -> None:
        # This also validates native construction and dataclasses.replace;
        # frozen fields alone do not guarantee a valid group homomorphism.
        if not isinstance(self.source, GAPGroup):
            raise GAPError("rho source must be a GAPGroup")
        for name, expected in (("generator_images", len(self.source.generators)), ("images", len(self.source.elements))):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise GAPError(f"rho {name} must be a sequence of integer bits")
            if len(values) != expected:
                raise GAPError(f"rho {name} requires {expected} integer bits")
            if any(type(bit) is not int or bit not in (0, 1) for bit in values):
                raise GAPError(f"rho {name} must contain only integer 0 or 1")
            object.__setattr__(self, name, tuple(values))
        if tuple(self(g) for g in self.source.generators) != self.generator_images:
            raise GAPError("rho images disagree with generator_images")
        if any(
            self(self.source.mul(g, h)) != (self.images[i] + self.images[j]) % 2
            for i, g in enumerate(self.source.elements) for j, h in enumerate(self.source.elements)
        ):
            raise GAPError("rho images violate the homomorphism law")

    def __call__(self, g: str) -> int:
        return self.images[self.source._position(g)]


@lru_cache(maxsize=128)
def _construct_group(
    expression: str, names: tuple[str, ...] | None, executable: str, timeout: float, max_order: int,
) -> GAPGroup:
    script = _group_setup(expression, max_order)
    script += f'Print("{_BEGIN}");\n' + _group_export() + f'Print("{_END}\\n");\nQUIT_GAP(0);\n'
    exported = _validate_export(_run_gap(script, executable, timeout), max_order)
    size, identity, generator_positions, table, inverses = exported
    if names is None:
        names = tuple(f"g{i + 1}" for i in range(len(generator_positions)))
    if len(names) != len(generator_positions):
        raise GAPError(f"GAP group has {len(generator_positions)} generators, but {len(names)} names were given")
    labels = {identity: "1"}
    order = [identity]
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for name, generator in zip(names, generator_positions):
            product = table[current][generator]
            if product in labels:
                continue
            labels[product] = name if current == identity else labels[current] + "*" + name
            order.append(product)
            queue.append(product)
    if len(order) != size:
        raise GAPError("GAP generators did not generate the exported group")
    elements = tuple(labels[index] for index in order)
    group = object.__new__(GAPGroup)
    values = dict(
        expression=expression, generator_names=names, elements=elements,
        generators=tuple(labels[index] for index in generator_positions),
        multiplication_table=tuple(tuple(labels[table[i][j]] for j in order) for i in order),
        identity="1",
        _inverses=tuple(labels[inverses[index]] for index in order),
        _positions=MappingProxyType({element: i for i, element in enumerate(elements)}),
        _gap_export=exported, _order=tuple(order), _executable=executable,
        _timeout=timeout, _max_order=max_order,
    )
    for name, value in values.items():
        object.__setattr__(group, name, value)
    return group


@lru_cache(maxsize=256)
def _construct_homomorphism(group: GAPGroup, bits: tuple[int, ...]) -> GAPZ2Homomorphism:
    script = _group_setup(group.expression, group._max_order)
    script += f"""
umtcZ2 := CyclicGroup(2);;
umtcZ2Generator := GeneratorsOfGroup(umtcZ2)[1];;
umtcBits := {json.dumps(bits)};;
umtcRho := GroupHomomorphismByImages(umtcG,umtcZ2,umtcGenerators,
    List(umtcBits,b -> umtcZ2Generator^b));;
if umtcRho = fail then Error("rho generator images do not define a homomorphism to Z2"); fi;
if not IsGroupHomomorphism(umtcRho) then Error("rho is not a group homomorphism"); fi;
Print("{_BEGIN}[");
"""
    script += _group_export()
    script += f"""
Print(",[",JoinStringsWithSeparator(List(umtcElements,function(x)
    if Image(umtcRho,x) = One(umtcZ2) then return "0"; else return "1"; fi;
end),","),"]]{_END}\\n");
QUIT_GAP(0);
"""
    raw = _run_gap(script, group._executable, group._timeout)
    if not isinstance(raw, list) or len(raw) != 2:
        raise GAPError("GAP returned an invalid rho response")
    exported = _validate_export(raw[0], group._max_order)
    if exported != group._gap_export:
        raise GAPError("GAP group enumeration changed while constructing rho")
    raw_images = raw[1]
    if not isinstance(raw_images, list) or len(raw_images) != len(group.elements) or any(
        type(bit) is not int or bit not in (0, 1) for bit in raw_images
    ):
        raise GAPError("GAP returned invalid rho images")
    images = tuple(raw_images[index] for index in group._order)
    return GAPZ2Homomorphism(group, bits, images)
