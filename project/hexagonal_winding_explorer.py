"""Interactive phase-winding explorer for a 2D reciprocal-lattice grid.

The array convention used throughout is::

    F[i1, i2]  <->  q(i1, i2)
                 = origin[0] * b1 + origin[1] * b2
                   + 2(i1 / N1) * b1 + 2(i2 / N2) * b2

by default.  Consequently, the complete displayed Q domain is the single
parallelogram spanned by ``2*b1`` and ``2*b2``.  No cells are stitched or
replicated.

Thus, a square loop in ``(i1, i2)`` is generally a parallelogram in Cartesian
momentum coordinates.  That is a perfectly valid contour for a winding
number.  The class displays the contour with its correct physical shape.

In JupyterLab, enable an interactive Matplotlib backend before constructing
the explorer, for example::

    %matplotlib widget

    explorer = InteractiveWindingExplorer(
        F[:, :, p1, p2],
        p=(p1, p2),
        config=cfg,
        # Or use lattice_type="hexagonal", b1=b1, b2=b2.
        radii=(2, 3, 4, 5),
        search_radius=5,
        boundary_mode="reject",   # safest default
        title=rf"$|F(Q,p_{{{p1},{p2}}})|$",
    )

Boundary modes
--------------
``reject``
    A contour that leaves the stored array is reported as invalid.  This is
    the safest choice if the boundary sewing law for F has not been supplied.
``periodic``
    Array indices are wrapped and F is assumed to be a periodic scalar.
``sewn``
    Array indices are wrapped and a user-supplied ``sewing_factor`` is used to
    construct the extended-zone value.

For ``boundary_mode="sewn"``, the callback has signature::

    sewing_factor(i1_wrapped, i2_wrapped, n1, n2) -> complex

where ``n1`` and ``n2`` count how many complete stored Q domains the
unwrapped index crossed.  With the default ``q_span=2``, those translations
are ``2*b1`` and ``2*b2``.  The sampled value is

    F_ext = sewing_factor(...) * F[i1_wrapped, i2_wrapped].

The callback must implement the sewing law appropriate to the particular
field F being inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


Index = Tuple[int, int]
SewingFactor = Callable[[int, int, int, int], complex]


def _periodic_fractional_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Maximum component distance on the fractional reciprocal-space torus."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.rint(delta)
    return float(np.max(np.abs(delta)))


def high_symmetry_label(
    fractional_momentum: Sequence[float],
    lattice_type: str,
    tol: float = 1e-8,
) -> Optional[str]:
    """Return the high-symmetry label of a momentum modulo b1 and b2."""
    if lattice_type == "hexagonal":
        points = {
            r"$\Gamma$": ([0, 0], [1, 0], [0, 1], [1, 1]),
            r"$K$": ([1 / 3, 1 / 3],),
            r"$K'$": ([2 / 3, 2 / 3],),
            r"$M$": ([1 / 2, 1 / 2],),
        }
    elif lattice_type == "square":
        points = {
            r"$\Gamma$": ([0, 0], [0, 1], [1, 0], [1, 1]),
            r"$X$": ([1 / 2, 0], [1 / 2, 1]),
            r"$X'$": ([0, 1 / 2], [1, 1 / 2]),
            r"$M$": ([1 / 2, 1 / 2],),
        }
    else:
        raise ValueError("lattice_type must be 'square' or 'hexagonal'")

    for label, equivalent_points in points.items():
        if any(
            _periodic_fractional_distance(fractional_momentum, point) <= tol
            for point in equivalent_points
        ):
            return label
    return None


def selected_particle_momenta(
    p_index: Sequence[int],
    Q_index: Index,
    grid: "ReciprocalGrid2D",
    lattice_type: str,
    Nk: Optional[Union[Sequence[float], float]] = None,
    symmetry_tol: float = 1e-8,
) -> dict[str, Any]:
    """Calculate electron/hole momenta for a selected Q-grid zero.

    The relative momentum is

        p = (p1/Nk1) b1 + (p2/Nk2) b2,

    while Q is taken from the explorer grid, including its origin and span.
    High-symmetry labels are identified modulo reciprocal vectors. Cartesian
    components are also returned in units of pi for display.
    """
    if p_index is None or len(p_index) < 2:
        raise ValueError("p_index must contain (p1, p2)")

    if Nk is None:
        Nk = (grid.N1 / grid.span[0], grid.N2 / grid.span[1])
    elif np.isscalar(Nk):
        Nk = (float(Nk), float(Nk))
    else:
        Nk = tuple(float(value) for value in Nk)
        if len(Nk) != 2:
            raise ValueError("Nk must be a number or a pair of numbers")
    if min(Nk) <= 0:
        raise ValueError("Nk must be positive")

    p_fractional = np.array(
        [float(p_index[0]) / Nk[0], float(p_index[1]) / Nk[1]]
    )
    Q_fractional = np.asarray(grid.index_to_fractional(*Q_index), dtype=float)
    electron_fractional = p_fractional + 0.5 * Q_fractional
    hole_fractional = p_fractional - 0.5 * Q_fractional

    def describe(fractional: np.ndarray) -> dict[str, Any]:
        cartesian = grid.fractional_to_cartesian(*fractional)
        label = high_symmetry_label(
            fractional, lattice_type=lattice_type, tol=symmetry_tol
        )
        return {
            "fractional": fractional,
            "cartesian": cartesian,
            "cartesian_over_pi": cartesian / np.pi,
            "label": label,
        }

    return {
        "p_fractional": p_fractional,
        "Q_fractional": Q_fractional,
        "electron": describe(electron_fractional),
        "hole": describe(hole_fractional),
    }


def format_particle_momentum(
    name: str,
    momentum: dict[str, Any],
    precision: int = 4,
) -> str:
    """Format a Cartesian momentum pair as multiples of pi."""
    kx, ky = momentum["cartesian_over_pi"]
    display_zero = 0.5 * 10.0 ** (-precision)
    kx = 0.0 if abs(kx) < display_zero else kx
    ky = 0.0 if abs(ky) < display_zero else ky
    label = f" ({momentum['label']})" if momentum["label"] else ""
    return (
        rf"${name}=({kx:.{precision}f}\pi,\ {ky:.{precision}f}\pi)$"
        + label
    )


def _plain_name(value: Any) -> Optional[str]:
    """Convert a string or Enum-like lattice/model value to lowercase text."""
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower().split(".")[-1]


def _config_value(config: Any, names: Sequence[str]) -> Any:
    """Read the first matching field from a mapping or config object."""
    if config is None:
        return None
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return None


def resolve_lattice_geometry(
    *,
    config: Any = None,
    lattice_type: Any = None,
    b1: Optional[Sequence[float]] = None,
    b2: Optional[Sequence[float]] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Resolve square/hexagonal geometry and reciprocal basis vectors.

    Priority is:

    1. Explicit ``b1`` and ``b2``.
    2. Vectors stored directly on/in ``config`` as ``b1`` and ``b2``.
    3. A 2x2 config field named ``reciprocal_vectors``, ``reciprocal_lattice``,
       ``bvecs``, or ``B`` (vectors are taken to be rows).
    4. A canonical dimensionless basis inferred from ``lattice_type`` or a
       config model/lattice field.

    The canonical basis is sufficient for topology and plot shape.  Supply
    physical b vectors if the Q-axis units and orientation must match the rest
    of a project exactly.
    """
    if (b1 is None) != (b2 is None):
        raise ValueError("supply both b1 and b2, or neither")

    config_b1 = _config_value(config, ("b1", "B1", "reciprocal_b1"))
    config_b2 = _config_value(config, ("b2", "B2", "reciprocal_b2"))
    if b1 is None and config_b1 is not None and config_b2 is not None:
        b1, b2 = config_b1, config_b2

    if b1 is None:
        vectors = _config_value(
            config,
            ("reciprocal_vectors", "reciprocal_lattice", "bvecs", "B"),
        )
        if vectors is not None:
            vectors = np.asarray(vectors, dtype=float)
            if vectors.shape != (2, 2):
                raise ValueError("config reciprocal vectors must have shape (2, 2)")
            b1, b2 = vectors[0], vectors[1]

    if lattice_type is None:
        lattice_type = _config_value(
            config,
            ("lattice_type", "ltype", "hamiltonian_type", "htype", "model"),
        )
    name = _plain_name(lattice_type)

    square_names = {"square", "orthogonal", "qwz", "qiwuzhang", "bhz"}
    hexagonal_names = {
        "hex", "hexagonal", "honeycomb", "triangular", "haldane", "graphene"
    }

    if name in square_names:
        kind = "square"
    elif name in hexagonal_names:
        kind = "hexagonal"
    elif name is None:
        if b1 is not None:
            test_b1 = np.asarray(b1, dtype=float)
            test_b2 = np.asarray(b2, dtype=float)
            norm_product = np.linalg.norm(test_b1) * np.linalg.norm(test_b2)
            cos_angle = abs(np.dot(test_b1, test_b2) / norm_product)
            equal_lengths = np.isclose(
                np.linalg.norm(test_b1), np.linalg.norm(test_b2), rtol=1e-5
            )
            kind = (
                "hexagonal"
                if equal_lengths and np.isclose(cos_angle, 0.5, atol=1e-5)
                else "square"
            )
        else:
            # Preserve the original explorer's behavior when no new geometry
            # input is supplied.
            kind = "square"
    else:
        raise ValueError(
            f"cannot infer lattice geometry from {lattice_type!r}; use "
            "lattice_type='square'/'hexagonal' or pass b1 and b2"
        )

    if b1 is None:
        if kind == "square":
            b1 = np.array([1.0, 0.0])
            b2 = np.array([0.0, 1.0])
        else:
            # A right-handed triangular reciprocal basis.  Overall rotation
            # and scale have no effect on the winding calculation.
            b1 = np.array([1.0, 0.0])
            b2 = np.array([0.5, np.sqrt(3.0) / 2.0])

    return kind, np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)


@dataclass(frozen=True)
class ReciprocalGrid2D:
    """Uniform 2D grid in a reciprocal parallelogram.

    ``span=(2, 2)`` means that the complete stored Q domain is spanned by
    ``2*b1`` and ``2*b2``.
    """

    b1: np.ndarray
    b2: np.ndarray
    shape: Tuple[int, int]
    origin: Tuple[float, float] = (0.0, 0.0)
    span: Tuple[float, float] = (2.0, 2.0)

    def __post_init__(self) -> None:
        b1 = np.asarray(self.b1, dtype=float)
        b2 = np.asarray(self.b2, dtype=float)
        if b1.shape != (2,) or b2.shape != (2,):
            raise ValueError("b1 and b2 must each contain two Cartesian components")
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError("shape must be a pair of integers, each at least 2")
        span = tuple(float(value) for value in self.span)
        if len(span) != 2 or min(span) <= 0:
            raise ValueError("span must contain two positive numbers")

        basis = np.column_stack((b1, b2))
        if abs(np.linalg.det(basis)) < 1e-14:
            raise ValueError("b1 and b2 must be linearly independent")

        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "span", span)

    @property
    def N1(self) -> int:
        return int(self.shape[0])

    @property
    def N2(self) -> int:
        return int(self.shape[1])

    @property
    def basis(self) -> np.ndarray:
        """Matrix whose columns are b1 and b2."""
        return np.column_stack((self.b1, self.b2))

    @property
    def orientation(self) -> int:
        """Sign of (b1 x b2)_z."""
        return 1 if np.linalg.det(self.basis) > 0 else -1

    def fractional_to_cartesian(self, q1, q2):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        return q1[..., None] * self.b1 + q2[..., None] * self.b2

    def cartesian_to_fractional(self, x: float, y: float) -> np.ndarray:
        return np.linalg.solve(self.basis, np.array([x, y], dtype=float))

    def index_to_fractional(self, i1, i2):
        q1 = self.origin[0] + self.span[0] * np.asarray(i1, dtype=float) / self.N1
        q2 = self.origin[1] + self.span[1] * np.asarray(i2, dtype=float) / self.N2
        return q1, q2

    def index_to_cartesian(self, i1, i2):
        q1, q2 = self.index_to_fractional(i1, i2)
        return self.fractional_to_cartesian(q1, q2)

    def center_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        i1, i2 = np.meshgrid(
            np.arange(self.N1), np.arange(self.N2), indexing="ij"
        )
        xy = self.index_to_cartesian(i1, i2)
        return xy[..., 0], xy[..., 1]

    def edge_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cell-edge coordinates suitable for pcolormesh(..., shading='flat')."""
        e1, e2 = np.meshgrid(
            np.arange(self.N1 + 1) - 0.5,
            np.arange(self.N2 + 1) - 0.5,
            indexing="ij",
        )
        xy = self.index_to_cartesian(e1, e2)
        return xy[..., 0], xy[..., 1]

    def nearest_unwrapped_index(self, x: float, y: float) -> Index:
        """Nearest grid index, without forcing it into the central cell.

        Rounding the two fractional components independently is not always the
        Euclidean-nearest operation on a skew grid.  We therefore check a 3x3
        neighborhood around that initial estimate using Cartesian distance.
        """
        q = self.cartesian_to_fractional(x, y)
        u1 = (q[0] - self.origin[0]) * self.N1 / self.span[0]
        u2 = (q[1] - self.origin[1]) * self.N2 / self.span[1]
        c1, c2 = int(np.rint(u1)), int(np.rint(u2))

        candidates = []
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                i1, i2 = c1 + d1, c2 + d2
                xy = self.index_to_cartesian(i1, i2)
                dist2 = (xy[0] - x) ** 2 + (xy[1] - y) ** 2
                candidates.append((dist2, i1, i2))
        _, i1, i2 = min(candidates, key=lambda item: item[0])
        return i1, i2

    def wrap_index(self, i1: int, i2: int) -> Tuple[int, int, int, int]:
        """Return wrapped indices and reciprocal-cell crossing counts."""
        n1, w1 = divmod(int(i1), self.N1)
        n2, w2 = divmod(int(i2), self.N2)
        return w1, w2, n1, n2

    def primitive_polygon(self) -> np.ndarray:
        x, y = self.edge_mesh()
        return np.array(
            [
                [x[0, 0], y[0, 0]],
                [x[-1, 0], y[-1, 0]],
                [x[-1, -1], y[-1, -1]],
                [x[0, -1], y[0, -1]],
            ]
        )


def _clip_polygon_to_halfplane(
    polygon: np.ndarray, normal: np.ndarray, offset: float, tol: float = 1e-12
) -> np.ndarray:
    """Clip a convex polygon to points satisfying dot(x, normal) <= offset."""
    if len(polygon) == 0:
        return polygon

    output = []
    previous = polygon[-1]
    previous_value = np.dot(previous, normal) - offset
    previous_inside = previous_value <= tol

    for current in polygon:
        current_value = np.dot(current, normal) - offset
        current_inside = current_value <= tol

        if current_inside != previous_inside:
            direction = current - previous
            denominator = np.dot(direction, normal)
            if abs(denominator) > tol:
                t = (offset - np.dot(previous, normal)) / denominator
                output.append(previous + t * direction)
        if current_inside:
            output.append(current)

        previous = current
        previous_inside = current_inside

    return np.asarray(output, dtype=float)


def wigner_seitz_polygon(b1, b2, neighbor_range: int = 3) -> np.ndarray:
    """Construct the reciprocal Wigner-Seitz cell by half-plane clipping.

    For a triangular reciprocal lattice the returned polygon has six vertices.
    The implementation also works for ordinary reduced 2D lattice bases.
    """
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    scale = max(np.linalg.norm(b1), np.linalg.norm(b2))
    size = 10.0 * neighbor_range * scale
    polygon = np.array(
        [[-size, -size], [size, -size], [size, size], [-size, size]],
        dtype=float,
    )

    reciprocal_vectors = []
    for n1 in range(-neighbor_range, neighbor_range + 1):
        for n2 in range(-neighbor_range, neighbor_range + 1):
            if n1 == 0 and n2 == 0:
                continue
            G = n1 * b1 + n2 * b2
            reciprocal_vectors.append(G)

    reciprocal_vectors.sort(key=lambda G: np.dot(G, G))
    for G in reciprocal_vectors:
        polygon = _clip_polygon_to_halfplane(
            polygon, normal=G, offset=0.5 * np.dot(G, G)
        )
        if len(polygon) == 0:
            raise RuntimeError("failed to construct the Wigner-Seitz polygon")

    center = polygon.mean(axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    return polygon[np.argsort(angles)]


def square_perimeter_indices(center: Index, radius: int) -> list[Index]:
    """Ordered perimeter of an index square, with each corner included once."""
    if radius < 1:
        raise ValueError("radius must be >= 1")

    c1, c2 = center
    low1, high1 = c1 - radius, c1 + radius
    low2, high2 = c2 - radius, c2 + radius
    points = []

    for i1 in range(low1, high1):
        points.append((i1, low2))
    for i2 in range(low2, high2):
        points.append((high1, i2))
    for i1 in range(high1, low1, -1):
        points.append((i1, high2))
    for i2 in range(high2, low2, -1):
        points.append((low1, i2))

    return points


@dataclass
class WindingResult:
    radius: int
    total_phase: float
    winding: float
    rounded_winding: Optional[int]
    min_abs_on_loop: float
    max_abs_phase_step: float
    valid: bool
    warning: str
    points: Optional[list[Index]]


def phase_winding_on_loop(
    values: Sequence[complex], zero_tol: float = 1e-12
) -> Tuple[float, float, float, float, str]:
    """Return total phase, winding, min |F|, max |dphi|, and a warning."""
    values = np.asarray(values, dtype=complex)
    if values.ndim != 1 or len(values) < 4:
        raise ValueError("values must be a one-dimensional loop with >= 4 samples")
    if not np.all(np.isfinite(values)):
        return np.nan, np.nan, np.nan, np.nan, "non-finite F on loop"

    magnitudes = np.abs(values)
    min_abs = float(np.min(magnitudes))
    reference = max(float(np.max(magnitudes)), 1.0)
    if min_abs <= zero_tol * reference:
        return (
            np.nan,
            np.nan,
            min_abs,
            np.nan,
            "loop passes through/too near a zero",
        )

    next_values = np.roll(values, -1)
    dphi = np.angle(next_values * np.conj(values))
    total = float(np.sum(dphi))
    winding = total / (2.0 * np.pi)
    max_step = float(np.max(np.abs(dphi)))

    warning = ""
    if max_step > 0.8 * np.pi:
        warning = "phase step near pi; refine grid or enlarge loop"
    elif abs(winding - np.rint(winding)) > 1e-6:
        warning = "winding is not numerically close to an integer"
    return total, winding, min_abs, max_step, warning


class ComplexFieldSampler:
    """Sample a stored field in reject, periodic, or sewn boundary mode."""

    VALID_MODES = {"reject", "periodic", "sewn"}

    def __init__(
        self,
        F: np.ndarray,
        boundary_mode: str = "reject",
        sewing_factor: Optional[SewingFactor] = None,
    ) -> None:
        self.F = np.asarray(F, dtype=complex)
        if self.F.ndim != 2:
            raise ValueError("F must be a two-dimensional complex array")
        if boundary_mode not in self.VALID_MODES:
            raise ValueError(f"boundary_mode must be one of {self.VALID_MODES}")
        if boundary_mode == "sewn" and sewing_factor is None:
            raise ValueError("boundary_mode='sewn' requires sewing_factor")
        self.boundary_mode = boundary_mode
        self.sewing_factor = sewing_factor

    def sample(self, i1: int, i2: int) -> complex:
        N1, N2 = self.F.shape
        inside = 0 <= i1 < N1 and 0 <= i2 < N2
        if self.boundary_mode == "reject":
            if not inside:
                raise IndexError("contour leaves stored reciprocal cell")
            return self.F[i1, i2]

        n1, w1 = divmod(int(i1), N1)
        n2, w2 = divmod(int(i2), N2)
        value = self.F[w1, w2]
        if self.boundary_mode == "sewn":
            value *= self.sewing_factor(w1, w2, n1, n2)
        return value


def phase_winding_square(
    sampler: ComplexFieldSampler,
    grid: ReciprocalGrid2D,
    center: Index,
    radius: int,
    zero_tol: float = 1e-12,
) -> WindingResult:
    """Compute winding on a physically counterclockwise index parallelogram."""
    points = square_perimeter_indices(center, radius)
    if grid.orientation < 0:
        points = points[::-1]

    try:
        values = [sampler.sample(i1, i2) for i1, i2 in points]
    except IndexError as error:
        return WindingResult(
            radius, np.nan, np.nan, None, np.nan, np.nan,
            False, str(error), None,
        )

    total, winding, min_abs, max_step, warning = phase_winding_on_loop(
        values, zero_tol=zero_tol
    )
    valid = bool(np.isfinite(winding))
    rounded = int(np.rint(winding)) if valid else None
    return WindingResult(
        radius, total, winding, rounded, min_abs, max_step,
        valid, warning, points,
    )


def find_local_minimum(
    sampler: ComplexFieldSampler,
    center: Index,
    search_radius: int,
    search_shape: str = "index_square",
    grid: Optional[ReciprocalGrid2D] = None,
) -> Index:
    """Find the smallest |F| near an unwrapped grid index.

    ``index_square`` checks all offsets in a square in fractional-grid space.
    ``cartesian_disk`` removes the square's distant Cartesian corners, using a
    disk whose radius is the larger of ``search_radius * |b1|/N1`` and
    ``search_radius * |b2|/N2``.
    """
    if search_radius < 0:
        raise ValueError("search_radius must be nonnegative")
    if search_shape not in {"index_square", "cartesian_disk"}:
        raise ValueError("search_shape must be 'index_square' or 'cartesian_disk'")
    if search_shape == "cartesian_disk" and grid is None:
        raise ValueError("cartesian_disk search requires grid")

    c1, c2 = center
    candidates = []
    if search_shape == "cartesian_disk":
        max_distance = search_radius * max(
            grid.span[0] * np.linalg.norm(grid.b1) / grid.N1,
            grid.span[1] * np.linalg.norm(grid.b2) / grid.N2,
        )

    for d1 in range(-search_radius, search_radius + 1):
        for d2 in range(-search_radius, search_radius + 1):
            if search_shape == "cartesian_disk":
                displacement = (
                    d1 * grid.span[0] * grid.b1 / grid.N1
                    + d2 * grid.span[1] * grid.b2 / grid.N2
                )
                if np.linalg.norm(displacement) > max_distance + 1e-14:
                    continue
            i1, i2 = c1 + d1, c2 + d2
            try:
                magnitude = abs(sampler.sample(i1, i2))
            except IndexError:
                continue
            candidates.append((magnitude, i1, i2))

    if not candidates:
        raise ValueError("search neighborhood contains no valid grid points")
    _, i1, i2 = min(candidates, key=lambda item: item[0])
    return i1, i2


class InteractiveWindingExplorer:
    """Click a zero of F and inspect windings on reciprocal-grid contours."""

    def __init__(
        self,
        F: np.ndarray,
        radii: Iterable[int] = (2, 3, 4, 5),
        search_radius: int = 5,
        p: Sequence[Any] = ("", ""),
        *,
        config: Any = None,
        lattice_type: Any = None,
        b1: Optional[Sequence[float]] = None,
        b2: Optional[Sequence[float]] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        q_span: Union[Sequence[float], float] = 2.0,
        search_shape: str = "index_square",
        display: str = "primitive",
        boundary_mode: str = "reject",
        sewing_factor: Optional[SewingFactor] = None,
        zero_tol: float = 1e-12,
        symmetry_tol: float = 1e-8,
        Nk: Optional[Union[Sequence[float], float]] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (12.0, 5.5),
        show_grid_points: bool = False,
    ) -> None:
        F = np.asarray(F)
        if F.ndim != 2:
            raise ValueError(
                "Pass a 2D slice, for example F[:, :, p1, p2], not the full array"
            )
        if display != "primitive":
            raise ValueError(
                "only display='primitive' is supported: the plot is the one "
                "stored Q-space parallelogram, without stitching or replication"
            )

        lattice_kind, resolved_b1, resolved_b2 = resolve_lattice_geometry(
            config=config, lattice_type=lattice_type, b1=b1, b2=b2
        )
        if np.isscalar(q_span):
            q_span = (float(q_span), float(q_span))
        else:
            q_span = tuple(float(value) for value in q_span)
            if len(q_span) != 2:
                raise ValueError("q_span must be a number or a pair of numbers")

        self.F = F.astype(complex, copy=False)
        self.A = np.abs(self.F)
        self.grid = ReciprocalGrid2D(
            resolved_b1, resolved_b2, self.F.shape,
            origin=origin, span=q_span,
        )
        self.sampler = ComplexFieldSampler(
            self.F, boundary_mode=boundary_mode, sewing_factor=sewing_factor
        )
        self.radii = [int(radius) for radius in radii]
        if not self.radii or min(self.radii) < 1:
            raise ValueError("radii must contain positive integers")
        self.search_radius = int(search_radius)
        self.search_shape = search_shape
        self.display = display
        self.lattice_kind = lattice_kind
        self.zero_tol = float(zero_tol)
        self.boundary_mode = boundary_mode
        self.p = p
        self.symmetry_tol = float(symmetry_tol)
        if Nk is None:
            Nk = _config_value(config, ("Nk", "nk", "N_k"))
        self.Nk = Nk

        self.fig, (self.ax_img, self.ax_txt) = plt.subplots(
            1, 2,
            figsize=figsize,
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.35, 1.0)},
        )
        self.loop_artists = []
        self._plot_field(cmap=cmap, show_grid_points=show_grid_points)

        if title is None:
            if p is not None and len(p) >= 2 and not (p[0] == "" and p[1] == ""):
                title = rf"$|F[:, :, {p[0]}, {p[1]}]|$"
            else:
                title = r"$|F(\mathbf{Q})|$"
        self.ax_img.set_title(title)
        self.ax_img.set_xlabel(r"$Q_x$")
        self.ax_img.set_ylabel(r"$Q_y$")
        self.ax_img.set_aspect("equal", adjustable="box")

        self.marker, = self.ax_img.plot(
            [], [], "o", ms=8, mfc="none", mec="red", mew=2,
            zorder=10, label="selected minimum",
        )

        self.ax_txt.set_axis_off()
        self.text = self.ax_txt.text(
            0.02, 0.98,
            "Click near a vortex/core",
            va="top", ha="left", family="monospace",
            transform=self.ax_txt.transAxes,
        )
        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick
        )

    def _plot_field(self, cmap: str, show_grid_points: bool) -> None:
        edge_x, edge_y = self.grid.edge_mesh()
        mesh = self.ax_img.pcolormesh(
            edge_x, edge_y, self.A, shading="flat", cmap=cmap
        )
        polygon = self.grid.primitive_polygon()
        self.ax_img.add_patch(
            Polygon(polygon, closed=True, fill=False, ec="white", lw=1.5)
        )
        self._set_limits_from_polygon(polygon)

        if show_grid_points:
            center_x, center_y = self.grid.center_mesh()
            self.ax_img.scatter(
                center_x, center_y, s=3, c="white", alpha=0.35, linewidths=0
            )

        self.mesh = mesh
        self.fig.colorbar(self.mesh, ax=self.ax_img, label=r"$|F|$")

    def _set_limits_from_polygon(self, polygon: np.ndarray) -> None:
        x_span = np.ptp(polygon[:, 0])
        y_span = np.ptp(polygon[:, 1])
        x_margin = 0.04 * max(x_span, 1e-15)
        y_margin = 0.04 * max(y_span, 1e-15)
        self.ax_img.set_xlim(polygon[:, 0].min() - x_margin,
                             polygon[:, 0].max() + x_margin)
        self.ax_img.set_ylim(polygon[:, 1].min() - y_margin,
                             polygon[:, 1].max() + y_margin)

    def clear_loops(self) -> None:
        for artist in self.loop_artists:
            artist.remove()
        self.loop_artists = []

    def draw_loop(self, points: Sequence[Index], **kwargs) -> None:
        i1 = np.array([point[0] for point in points] + [points[0][0]])
        i2 = np.array([point[1] for point in points] + [points[0][1]])
        xy = self.grid.index_to_cartesian(i1, i2)
        artist, = self.ax_img.plot(xy[:, 0], xy[:, 1], **kwargs)
        self.loop_artists.append(artist)

    def analyze(self, center: Index) -> list[WindingResult]:
        return [
            phase_winding_square(
                self.sampler, self.grid, center, radius,
                zero_tol=self.zero_tol,
            )
            for radius in self.radii
        ]

    def onclick(self, event) -> None:
        if event.inaxes is not self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        clicked_index = self.grid.nearest_unwrapped_index(
            event.xdata, event.ydata
        )
        try:
            center = find_local_minimum(
                self.sampler,
                clicked_index,
                self.search_radius,
                search_shape=self.search_shape,
                grid=self.grid,
            )
        except ValueError as error:
            self.text.set_text(f"Selection failed:\n{error}")
            self.fig.canvas.draw_idle()
            return

        xy = self.grid.index_to_cartesian(*center)
        self.marker.set_data([xy[0]], [xy[1]])
        self.clear_loops()

        w1, w2, n1, n2 = self.grid.wrap_index(*center)
        selected_abs = abs(self.sampler.sample(*center))
        lines = [
            f"unwrapped index: ({center[0]}, {center[1]})",
            f"stored index:    ({w1}, {w2})",
            f"cell shift:      ({n1}, {n2})",
            f"|F| there:       {selected_abs:.6g}",
            f"boundary mode:   {self.boundary_mode}",
        ]

        if (
            self.p is not None
            and len(self.p) >= 2
            and self.p[0] != ""
            and self.p[1] != ""
        ):
            momenta = selected_particle_momenta(
                self.p,
                center,
                self.grid,
                self.lattice_kind,
                Nk=self.Nk,
                symmetry_tol=self.symmetry_tol,
            )
            lines.extend([
                format_particle_momentum(r"k_e", momenta["electron"]),
                format_particle_momentum(r"k_h", momenta["hole"]),
            ])

        lines.extend([
            "",
            " r     winding  round    min|F|   max|dphi|  status",
        ])

        colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(self.radii)))
        for result, color in zip(self.analyze(center), colors):
            if result.points is not None:
                self.draw_loop(result.points, color=color, lw=1.6, zorder=8)

            winding_text = (
                f"{result.winding:9.5f}" if result.valid else "      nan"
            )
            rounded_text = (
                f"{result.rounded_winding:5d}"
                if result.rounded_winding is not None else "  nan"
            )
            min_text = (
                f"{result.min_abs_on_loop:9.2e}"
                if np.isfinite(result.min_abs_on_loop) else "      nan"
            )
            step_text = (
                f"{result.max_abs_phase_step:10.3f}"
                if np.isfinite(result.max_abs_phase_step) else "       nan"
            )
            status = result.warning or "ok"
            lines.append(
                f"{result.radius:2d}  {winding_text}  {rounded_text}  "
                f"{min_text}  {step_text}  {status}"
            )

        self.text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def disconnect(self) -> None:
        """Disconnect the click callback without closing the figure."""
        self.fig.canvas.mpl_disconnect(self.cid)


if __name__ == "__main__":
    # Minimal noninteractive demo.  Run in an interactive backend and click
    # near the center to test the explorer.
    N = 51
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    i1, i2 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    center = N // 2
    F_demo = (i1 - center) + 1j * (i2 - center)

    explorer = InteractiveWindingExplorer(
        F_demo,
        radii=(2, 3, 4, 5),
        search_radius=5,
        lattice_type="hexagonal",
        b1=b1,
        b2=b2,
        origin=(-0.5, -0.5),
        display="primitive",
        boundary_mode="reject",
        title="Synthetic +1 vortex on a triangular reciprocal grid",
    )
    plt.show()
