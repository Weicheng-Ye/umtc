"""Independent multiplicity fixture from concrete representations of A4.

Rep(A4) is a unitary symmetric fusion category, not a UMTC.  Its object ``v``
obeys v*v = 1+w+w2+2v, so it exercises genuine fusion multiplicities while
providing an intentional negative example for the modularity check.  These
symbols are obtained from tensor-product intertwiners, not coherence equations.
"""

import cmath
import copy
import itertools
import math
import unittest

from umtc import (
    UMTC,
    check_fusion,
    check_hexagon,
    check_modularity,
    check_pentagon,
    check_ribbon,
    check_symmetry,
    check_symmetry_f,
    check_symmetry_r,
    check_u_consistency,
    check_unitarity,
)


_OBJECTS = ("1", "w", "w2", "v")
_DIMENSIONS = {"1": 1, "w": 1, "w2": 1, "v": 3}
_CHARACTERS = {"1": 0, "w": 1, "w2": 2}
_OMEGA = cmath.exp(2j * math.pi / 3)


def _zero(rows, columns):
    return [[0j for _ in range(columns)] for _ in range(rows)]


def _cg_embeddings():
    """Return C_ab^(c,mu): H_c -> H_a tensor H_b as rectangular arrays."""
    embeddings = {}
    for a, b in itertools.product(_OBJECTS, repeat=2):
        if a != "v" and b != "v":
            c = _OBJECTS[(_CHARACTERS[a] + _CHARACTERS[b]) % 3]
            embeddings[a, b, c] = [[[1 + 0j]]]
        elif a != "v" or b != "v":
            k = _CHARACTERS[a if a != "v" else b]
            matrix = _zero(3, 3)
            for i in range(3):
                matrix[i][i] = _OMEGA ** (k * i)
            embeddings[a, b, "v"] = [matrix]
        else:
            for c, k in _CHARACTERS.items():
                matrix = _zero(9, 1)
                for i in range(3):
                    matrix[3 * i + i][0] = _OMEGA ** (-k * i) / math.sqrt(3)
                embeddings[a, b, c] = [matrix]
            symmetric, antisymmetric = _zero(9, 3), _zero(9, 3)
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                symmetric[3 * j + k][i] = 1 / math.sqrt(2)
                symmetric[3 * k + j][i] = 1 / math.sqrt(2)
                antisymmetric[3 * j + k][i] = 1 / math.sqrt(2)
                antisymmetric[3 * k + j][i] = -1 / math.sqrt(2)
            # A nonreal unitary change of multiplicity basis ensures that the
            # R block is not diagonal or symmetric: its offdiagonals are ±i.
            embeddings[a, b, "v"] = [
                [
                    [(symmetric[r][s] + 1j * antisymmetric[r][s]) / math.sqrt(2)
                     for s in range(3)]
                    for r in range(9)
                ],
                [
                    [(1j * symmetric[r][s] + antisymmetric[r][s]) / math.sqrt(2)
                     for s in range(3)]
                    for r in range(9)
                ],
            ]
    return embeddings


def _generator(a, name):
    """Generators P (order 3) and D (a double transposition) of A4."""
    if a != "v":
        return [[_OMEGA ** _CHARACTERS[a] if name == "P" else 1 + 0j]]
    result = _zero(3, 3)
    for i in range(3):
        if name == "P":
            result[(i + 1) % 3][i] = 1
        else:
            result[i][i] = 1 if i == 0 else -1
    return result


def _left_embedding(ab, ec, da, db, dc, de, dd):
    """Compose (C_ab^e tensor I_c) C_ec^d."""
    result = _zero(da * db * dc, dd)
    for ia, ib, ic, target in itertools.product(
        range(da), range(db), range(dc), range(dd)
    ):
        result[(ia * db + ib) * dc + ic][target] = sum(
            ab[ia * db + ib][intermediate] * ec[intermediate * dc + ic][target]
            for intermediate in range(de)
        )
    return result


def _right_embedding(bc, af, da, db, dc, df, dd):
    """Compose (I_a tensor C_bc^f) C_af^d."""
    result = _zero(da * db * dc, dd)
    for ia, ib, ic, target in itertools.product(
        range(da), range(db), range(dc), range(dd)
    ):
        result[(ia * db + ib) * dc + ic][target] = sum(
            bc[ib * dc + ic][intermediate] * af[ia * df + intermediate][target]
            for intermediate in range(df)
        )
    return result


def _overlap(destination, source, target_dimension):
    # Schur's lemma makes destination^dagger source a scalar identity.
    return sum(
        destination[i][j].conjugate() * source[i][j]
        for i in range(len(source))
        for j in range(target_dimension)
    ) / target_dimension


def _json_complex(value):
    return {"re": value.real, "im": value.imag}


def _a4_data():
    cg = _cg_embeddings()
    data = {
        "schema_version": 1,
        "name": "Rep(A4) multiplicity regression fixture",
        "anyons": list(_OBJECTS),
        "vacuum": "1",
        "fusion_rules": [
            {"a": a, "b": b, "c": c, "multiplicity": len(matrices)}
            for (a, b, c), matrices in cg.items()
        ],
        "quantum_dimensions": [
            {"anyon": a, "value": _DIMENSIONS[a]} for a in _OBJECTS
        ],
        "topological_spins": [{"anyon": a, "value": 1} for a in _OBJECTS],
        "F_symbols": [],
        "R_symbols": [],
        "symmetry": None,
    }
    for a, b, c, d in itertools.product(_OBJECTS, repeat=4):
        da, db, dc, dd = (_DIMENSIONS[x] for x in (a, b, c, d))
        left, right = [], []
        for intermediate in _OBJECTS:
            di = _DIMENSIONS[intermediate]
            for ab, ec in itertools.product(
                cg.get((a, b, intermediate), []), cg.get((intermediate, c, d), [])
            ):
                left.append(_left_embedding(ab, ec, da, db, dc, di, dd))
            for bc, af in itertools.product(
                cg.get((b, c, intermediate), []), cg.get((a, intermediate, d), [])
            ):
                right.append(_right_embedding(bc, af, da, db, dc, di, dd))
        if not left:
            continue
        matrix = [
            [_json_complex(_overlap(r, l, dd)) for r in right] for l in left
        ]
        data["F_symbols"].append(
            {"a": a, "b": b, "c": c, "d": d, "matrix": matrix}
        )
    for (a, b, c), ab in cg.items():
        da, db, dc = (_DIMENSIONS[x] for x in (a, b, c))
        swapped = [
            [matrix[ib * da + ia][:] for ia in range(da) for ib in range(db)]
            for matrix in cg[b, a, c]
        ]
        # Paper Eq. (9): rows are the ba vertex before the crossing; columns
        # are the ab vertex after it.  This overlap fixes that convention.
        matrix = [
            [_json_complex(_overlap(destination, source, dc)) for destination in ab]
            for source in swapped
        ]
        data["R_symbols"].append({"a": a, "b": b, "c": c, "matrix": matrix})
    return data


def _a4_symmetry_data(base):
    """Add commuting abstract symmetries with noncommuting U matrices.

    In the unrotated CG basis, X is induced by the coordinate reflection
    e_i -> e_(-i), while T is complex conjugation.  Both exchange w and w2.
    U_X is diag(1,-1) on the two v*v->v vertices, and U_T is identity.
    Applying the same complex multiplicity-basis change as in _cg_embeddings
    gives the matrices below.  Antiunitary conjugation reconciles their
    noncommutativity with the abstract group X*T = T*X.
    """
    data = copy.deepcopy(base)
    groups = ("1", "X", "T", "XT")
    blocks = {
        "1": [[1, 0], [0, 1]],
        "X": [[0, -1j], [1j, 0]],
        "T": [[0, -1j], [-1j, 0]],
        "XT": [[1, 0], [0, -1]],
    }
    data["symmetry"] = {
        "elements": list(groups),
        "identity": "1",
        "multiplication_table": [[groups[g ^ h] for h in range(4)] for g in range(4)],
        "antiunitary": ["T", "XT"],
        "action": [
            {"g": g, "images": ["1", "w2", "w", "v"] if g in ("X", "T") else list(_OBJECTS)}
            for g in groups
        ],
        "U_symbols": [
            {
                "g": g, "a": row["a"], "b": row["b"], "c": row["c"],
                "matrix": [
                    [_json_complex(complex(value)) for value in values]
                    for values in (blocks[g] if (row["a"], row["b"], row["c"]) == ("v",) * 3 else [[1]])
                ],
            }
            for g in groups for row in data["fusion_rules"]
        ],
        "eta_symbols": [
            {"anyon": a, "g": g, "h": h, "value": 1}
            for a, g, h in itertools.product(_OBJECTS, groups, groups)
        ],
    }
    return data


class MultiplicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _a4_data()
        cls.category = UMTC.from_dict(cls.data)

    def test_clebsch_gordan_maps_are_complete_isometric_intertwiners(self):
        """Validate the concrete representation input without any F/R checks."""
        cg = _cg_embeddings()
        for a, b in itertools.product(_OBJECTS, repeat=2):
            da, db = _DIMENSIONS[a], _DIMENSIONS[b]
            columns = [
                [matrix[row][column] for row in range(da * db)]
                for c in _OBJECTS
                for matrix in cg.get((a, b, c), [])
                for column in range(_DIMENSIONS[c])
            ]
            self.assertEqual(len(columns), da * db)
            for i, left in enumerate(columns):
                for j, right in enumerate(columns):
                    overlap = sum(x.conjugate() * y for x, y in zip(left, right))
                    self.assertLess(abs(overlap - (i == j)), 1e-12)
        for (a, b, c), matrices in cg.items():
            da, db, dc = (_DIMENSIONS[x] for x in (a, b, c))
            for name in ("P", "D"):
                ga, gb, gc = (_generator(x, name) for x in (a, b, c))
                for matrix in matrices:
                    for ia, ib, target in itertools.product(
                        range(da), range(db), range(dc)
                    ):
                        left = sum(
                            ga[ia][ja] * gb[ib][jb] * matrix[ja * db + jb][target]
                            for ja in range(da) for jb in range(db)
                        )
                        right = sum(
                            matrix[ia * db + ib][j] * gc[j][target] for j in range(dc)
                        )
                        self.assertLess(abs(left - right), 1e-12)

    def test_coherence_with_nonreal_offdiagonal_multiplicity_braiding(self):
        block = next(
            entry for entry in self.data["R_symbols"]
            if (entry["a"], entry["b"], entry["c"]) == ("v", "v", "v")
        )["matrix"]
        self.assertEqual(len(block), 2)
        self.assertGreater(abs(block[0][1]["im"]), 0.9)
        self.assertLess(abs(block[0][1]["im"] + block[1][0]["im"]), 1e-12)
        for check in (check_fusion, check_unitarity, check_pentagon, check_hexagon, check_ribbon):
            with self.subTest(check=check.__name__):
                self.assertTrue(check(self.category))
        self.assertFalse(check_modularity(self.category))

    def test_corrupted_multiplicity_fails_pentagon(self):
        data = copy.deepcopy(self.data)
        block = next(
            entry for entry in data["F_symbols"]
            if (entry["a"], entry["b"], entry["c"], entry["d"]) == ("v",) * 4
        )["matrix"]
        self.assertEqual(len(block), 7)
        # Channel indices 3..6 are the two independent v vertices on each leg.
        block[3][4]["re"] += 0.125
        self.assertFalse(check_pentagon(UMTC.from_dict(data)))

    def test_corrupted_multiplicity_fails_hexagon(self):
        data = copy.deepcopy(self.data)
        block = next(
            entry for entry in data["R_symbols"]
            if (entry["a"], entry["b"], entry["c"]) == ("v", "v", "v")
        )["matrix"]
        block[0][1]["re"] += 0.125
        self.assertFalse(check_hexagon(UMTC.from_dict(data)))

    def test_noncommuting_multiplicity_u_matrices_obey_antiunitary_composition(self):
        category = UMTC.from_dict(_a4_symmetry_data(self.data))
        symmetry = category.symmetry
        ux = symmetry.U_matrix("X", "v", "v", "v")
        ut = symmetry.U_matrix("T", "v", "v", "v")
        xt = sum(ux[0][k] * ut[k][0] for k in range(2))
        tx = sum(ut[0][k] * ux[k][0] for k in range(2))
        self.assertGreater(abs(xt - tx), 1.9)
        self.assertTrue(check_symmetry(category))

    def test_wrong_u_product_order_or_transpose_fails_composition(self):
        for corruption in ("product_order", "transpose"):
            with self.subTest(corruption=corruption):
                data = _a4_symmetry_data(self.data)
                group = "XT" if corruption == "product_order" else "X"
                entry = next(
                    row for row in data["symmetry"]["U_symbols"]
                    if (row["g"], row["a"], row["b"], row["c"]) == (group, "v", "v", "v")
                )
                if corruption == "product_order":
                    # U_X U_T = -U_T U_X, whereas Eq. (23) requires U_T U_X.
                    for row in entry["matrix"]:
                        for cell in row:
                            cell["re"] *= -1
                            cell["im"] *= -1
                else:
                    entry["matrix"] = [list(row) for row in zip(*entry["matrix"])]
                category = UMTC.from_dict(data)
                # These wrong choices still define individual braided functors;
                # only compatibility with the supplied group composition fails.
                self.assertTrue(check_symmetry_f(category))
                self.assertTrue(check_symmetry_r(category))
                self.assertFalse(check_u_consistency(category))
                self.assertFalse(check_symmetry(category))


if __name__ == "__main__":
    unittest.main()
