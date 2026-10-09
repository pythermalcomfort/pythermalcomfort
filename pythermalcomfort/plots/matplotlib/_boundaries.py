"""Root-finding backend for threshold-region boundaries.

Rasterising the whole plane and contouring it makes every edge out of grid
cells, so region fills and the out-of-limits area come out as staircases whose
step size is the grid resolution.

This module finds boundaries directly instead.  For every row of the plot it
locates the *exact* position of each boundary by bisection, so the result is a
set of ``(x, y)`` polylines that can be drawn with ``ax.plot`` and filled with
``ax.fill_betweenx``.  The step size no longer depends on the grid: it depends
on the bisection tolerance, six orders of magnitude finer.

Two kinds of boundary are solved the same way, by bisecting a boolean
predicate along one axis:

``threshold``
    where the model output crosses a threshold; the predicate is
    ``output >= threshold``.
``validity``
    where the model leaves its applicability limits and starts returning NaN;
    the predicate is ``isfinite(output)``.

Bands, not regions
------------------

A threshold may be crossed more than once along a row.  ``ppd`` is the usual
example: it falls to a minimum at neutrality and rises again, so a row crosses
``ppd = 10`` twice and the "below 10" region is a single strip with "above 10"
on *both* sides.

So the solver does not work in regions.  It cuts each row at every crossing and
calls the resulting slices **bands**.  Each band carries the index of the
region it belongs to, and several bands can share one region.  A monotone
output simply produces one band per region, the ordinary case.

For this to be fillable, the layout has to be the same in every row: the same
thresholds crossed in the same order, running the same way.  A threshold that
is crossed in some rows and not others is fine -- the missing crossing
collapses onto the edge of the valid area and its band becomes zero-width,
which is what happens when a comfort zone runs off the side of the chart.
Geometry that cannot be laid out consistently -- an internal hole in the valid
area, or a pair of crossings that appears only halfway up the chart -- is
reported by returning ``None``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

#: Bisection stops once every bracket is narrower than this fraction of the
#: scanned axis span.  1e-6 of a 16 degC span is 16 microkelvin, far below the
#: width of a plotted line.
_REL_TOL: float = 1e-6

#: Hard cap on bisection iterations.  Each iteration halves every bracket, so
#: 60 steps take any starting bracket below double precision.
_MAX_BISECT_STEPS: int = 60

#: Signature of the model-evaluation callback: takes flat ``scan`` and ``row``
#: coordinate arrays of equal length, returns the model output for each pair.
EvaluateFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _ValidityEdge(NamedTuple):
    """Where a row's valid interval begins or ends.

    Attributes
    ----------
    row : int
        Row the edge belongs to.
    high_end : bool
        ``True`` for the high end of the scan axis, ``False`` for the low end.
    """

    row: int
    high_end: bool


class _Cut(NamedTuple):
    """Where a row crosses one branch of one threshold.

    Attributes
    ----------
    threshold : int
        Index into the threshold list.
    branch : int
        Which crossing of that threshold, counted along the scan axis.
    row : int
        Row the crossing belongs to.
    """

    threshold: int
    branch: int
    row: int


class _Bracket(NamedTuple):
    """One boundary located well enough to bisect.

    Attributes
    ----------
    lo, hi : float
        Scan coordinates straddling the boundary; ``predicate`` differs
        between them.
    row : int
        Row being scanned.
    target : float
        Threshold the predicate compares against, ``NaN`` for a validity edge
        whose predicate is ``isfinite`` instead.
    predicate_at_lo : bool
        Value of the predicate at ``lo``.
    slot : _ValidityEdge or _Cut
        What the refined position is for.
    """

    lo: float
    hi: float
    row: int
    target: float
    predicate_at_lo: bool
    slot: _ValidityEdge | _Cut


@dataclass(frozen=True)
class BoundaryCurve:
    """One branch of one threshold boundary, as a polyline in data coordinates.

    A threshold crossed twice per row yields two curves, ``branch`` 0 and 1.

    Attributes
    ----------
    threshold : float
        The output value this curve traces.
    branch : int
        Which crossing of that threshold this curve follows, counted along the
        scanned axis.  Always ``0`` for a monotone output.
    x : numpy.ndarray
        X coordinates, with ``NaN`` wherever the threshold is not actually
        crossed, which breaks the curve into segments when plotted.
    y : numpy.ndarray
        Y coordinates, same length as ``x``.
    """

    threshold: float
    branch: int
    x: np.ndarray
    y: np.ndarray


@dataclass
class RegionBands:
    """Bands solved row by row along one axis.

    Attributes
    ----------
    scan_axis : str
        ``"x"`` if boundaries were solved by scanning along x for each y,
        ``"y"`` the other way round.
    rows : numpy.ndarray
        Coordinates along the axis *perpendicular* to the scan, one per row,
        shape ``(n_rows,)``.
    edges : numpy.ndarray
        Band boundaries along the scan axis, shape ``(n_bands + 1, n_rows)``.
        Band ``j`` spans ``edges[j]`` to ``edges[j + 1]``; consecutive edges
        are equal wherever a band is empty in that row.
    band_regions : list of int
        Region index of each band, length ``n_bands``.  Several bands may
        share a region.
    valid_start, valid_end : numpy.ndarray
        Extent of the model's valid area along the scan axis, shape
        ``(n_rows,)``.  Rows with no valid point at all collapse both to the
        low end of the scan axis, so the caller can shade the whole row as
        out-of-limits.
    has_valid : numpy.ndarray
        Boolean, shape ``(n_rows,)``, False for rows with no valid point.
    has_invalid : bool
        True when any part of the plotted area falls outside the model's
        applicability limits.
    curves : list of BoundaryCurve
        One curve per threshold per branch, ordered by threshold then branch.
    """

    scan_axis: str
    rows: np.ndarray
    edges: np.ndarray
    band_regions: list[int]
    valid_start: np.ndarray
    valid_end: np.ndarray
    has_valid: np.ndarray
    has_invalid: bool
    curves: list[BoundaryCurve] = field(default_factory=list)


def _bisect(
    *,
    evaluate: EvaluateFn,
    lo: np.ndarray,
    hi: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    predicate_at_lo: np.ndarray,
    tol: float,
) -> np.ndarray | None:
    """Locate, by vectorised bisection, where each bracket's predicate flips.

    Every bracket carries its own predicate: ``isfinite(output)`` when its
    entry in ``targets`` is ``NaN``, otherwise ``output >= target``.  All
    brackets advance together, so the whole set costs one model call per
    iteration rather than one per bracket.

    Parameters
    ----------
    evaluate : callable
        Model-evaluation callback.
    lo, hi : numpy.ndarray
        Bracket endpoints along the scan axis; the predicate differs between
        them.
    rows : numpy.ndarray
        Row coordinate of each bracket.
    targets : numpy.ndarray
        Threshold for each bracket, ``NaN`` for validity brackets.
    predicate_at_lo : numpy.ndarray
        Boolean value of each bracket's predicate at ``lo``.
    tol : float
        Stop once every bracket is narrower than this.

    Returns
    -------
    numpy.ndarray or None
        Midpoint of each converged bracket, or ``None`` if the model returned
        NaN inside a threshold bracket, which means the bracket straddles an
        out-of-limits pocket and the answer would be meaningless.
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    is_validity = np.isnan(targets)

    for _ in range(_MAX_BISECT_STEPS):
        if np.all(np.abs(hi - lo) <= tol):
            break
        mid = 0.5 * (lo + hi)
        z_mid = np.asarray(evaluate(mid, rows), dtype=float).ravel()
        finite_mid = np.isfinite(z_mid)

        if np.any(~is_validity & ~finite_mid):
            return None

        with np.errstate(invalid="ignore"):
            above = z_mid >= targets
        predicate_at_mid = np.where(is_validity, finite_mid, above)

        keep_low = predicate_at_mid == predicate_at_lo
        lo = np.where(keep_low, mid, lo)
        hi = np.where(keep_low, hi, mid)

    return 0.5 * (lo + hi)


@dataclass
class _RowScan:
    """What one pass over the grid learned, before any refinement.

    Attributes
    ----------
    brackets : list of _Bracket
        One per boundary still to be refined.
    valid_start, valid_end : numpy.ndarray
        Extent of the valid interval per row, ``NaN`` where a bracket still
        has to be refined.
    has_valid : numpy.ndarray
        Rows with at least one valid point.
    starts_above : numpy.ndarray
        Whether each row's first valid sample is at or above each threshold,
        shape ``(n_thresholds, n_rows)``.
    directions : list of int
        ``+1`` where the output rises through a crossing along the scan axis,
        ``-1`` where it falls.  Flat, indexed by :func:`_cut_index`.
    n_crossings : list of int
        How many times each threshold is crossed per row.
    """

    brackets: list[_Bracket]
    valid_start: np.ndarray
    valid_end: np.ndarray
    has_valid: np.ndarray
    starts_above: np.ndarray
    directions: list[int]
    n_crossings: list[int]


def _cut_index(n_crossings: Sequence[int], threshold: int, crossing: int) -> int:
    """Flat index of one crossing within a row's cut list."""
    return sum(n_crossings[:threshold]) + crossing


def _canonical_branches(
    flips: list[list[list[tuple[int, int]]]],
    *,
    has_valid: np.ndarray,
) -> list[tuple[int, ...]] | None:
    """Work out the full set of crossings each threshold makes.

    The row that crosses a threshold the most times shows its complete
    structure; rows crossing it fewer times have had a branch clipped off by
    the edge of the valid area.  Every row that shows the full structure has to
    agree on it.

    Parameters
    ----------
    flips : list
        ``flips[threshold][row]`` is a list of ``(sample index, direction)``
        for each crossing found in that row.
    has_valid : numpy.ndarray
        Rows with at least one valid point.

    Returns
    -------
    list of tuple or None
        Direction of each branch, per threshold; ``None`` if the rows showing
        the full structure disagree about it.
    """
    canonical: list[tuple[int, ...]] = []
    for per_row in flips:
        best: tuple[int, ...] = ()
        for row, found in enumerate(per_row):
            if not has_valid[row]:
                continue
            directions = tuple(direction for _, direction in found)
            if len(directions) > len(best):
                best = directions
            elif len(directions) == len(best) and directions != best:
                return None
        canonical.append(best)
    return canonical


def _align_to_branches(
    found: list[tuple[int, int]], canonical: tuple[int, ...]
) -> list[int] | None:
    """Match a row's crossings onto the canonical branches, by direction.

    A row missing a branch is a row where that side of the region ran past the
    edge of the valid area.  Walking both lists left to right and pairing on
    direction says which branch each surviving crossing is.

    Parameters
    ----------
    found : list of tuple
        ``(sample index, direction)`` per crossing in this row.
    canonical : tuple of int
        Direction of every branch of this threshold.

    Returns
    -------
    list of int or None
        Branch index for each crossing, or ``None`` if they cannot be matched.
    """
    branch = 0
    assignment: list[int] = []
    for _, direction in found:
        while branch < len(canonical) and canonical[branch] != direction:
            branch += 1
        if branch == len(canonical):
            return None
        assignment.append(branch)
        branch += 1
    return assignment


def _scan_rows(
    *,
    grid: np.ndarray,
    scan: np.ndarray,
    thresholds: Sequence[float],
) -> _RowScan | None:
    """Analyse each row of the evaluated grid and collect bisection brackets.

    Parameters
    ----------
    grid : numpy.ndarray
        Model output, shape ``(n_rows, n_scan)``.
    scan : numpy.ndarray
        Coordinates along the scan axis, shape ``(n_scan,)``.
    thresholds : sequence of float
        Threshold values, ascending.

    Returns
    -------
    _RowScan or None
        ``None`` when a row's valid area is not contiguous, or when the rows
        disagree about how many times a threshold is crossed and which way.
    """
    n_rows = grid.shape[0]
    n_thresholds = len(thresholds)
    finite = np.isfinite(grid)

    valid_start = np.full(n_rows, np.nan)
    valid_end = np.full(n_rows, np.nan)
    has_valid = np.zeros(n_rows, dtype=bool)
    starts_above = np.zeros((n_thresholds, n_rows), dtype=bool)
    runs: dict[int, tuple[int, int]] = {}
    flips: list[list[list[tuple[int, int]]]] = [
        [[] for _ in range(n_rows)] for _ in range(n_thresholds)
    ]
    brackets: list[_Bracket] = []

    for row in range(n_rows):
        row_finite = finite[row]
        if not row_finite.any():
            continue

        indices = np.flatnonzero(row_finite)
        first, last = int(indices[0]), int(indices[-1])
        if last - first + 1 != indices.size:
            # A hole in the middle of the valid area splits the row in two.
            return None

        has_valid[row] = True
        runs[row] = (first, last)
        z_run = grid[row, first : last + 1]

        if first == 0:
            valid_start[row] = scan[0]
        else:
            brackets.append(
                _Bracket(
                    lo=scan[first - 1],
                    hi=scan[first],
                    row=row,
                    target=np.nan,
                    predicate_at_lo=False,
                    slot=_ValidityEdge(row=row, high_end=False),
                )
            )
        if last == scan.size - 1:
            valid_end[row] = scan[-1]
        else:
            brackets.append(
                _Bracket(
                    lo=scan[last],
                    hi=scan[last + 1],
                    row=row,
                    target=np.nan,
                    predicate_at_lo=True,
                    slot=_ValidityEdge(row=row, high_end=True),
                )
            )

        for k, threshold in enumerate(thresholds):
            above = z_run >= threshold
            starts_above[k, row] = bool(above[0])
            flips[k][row] = [
                (int(flip), 1 if above[flip + 1] else -1)
                for flip in np.flatnonzero(above[:-1] != above[1:])
            ]

    if not has_valid.any():
        return None

    canonical = _canonical_branches(flips, has_valid=has_valid)
    if canonical is None:
        return None

    for row, (first, last) in runs.items():
        s_run = scan[first : last + 1]
        z_run = grid[row, first : last + 1]
        for k, threshold in enumerate(thresholds):
            assignment = _align_to_branches(flips[k][row], canonical[k])
            if assignment is None:
                return None
            for (flip, _), branch in zip(flips[k][row], assignment, strict=True):
                brackets.append(
                    _Bracket(
                        lo=s_run[flip],
                        hi=s_run[flip + 1],
                        row=row,
                        target=float(threshold),
                        predicate_at_lo=bool(z_run[flip] >= threshold),
                        slot=_Cut(threshold=k, branch=branch, row=row),
                    )
                )

    return _RowScan(
        brackets=brackets,
        valid_start=valid_start,
        valid_end=valid_end,
        has_valid=has_valid,
        starts_above=starts_above,
        directions=[d for per_threshold in canonical for d in per_threshold],
        n_crossings=[len(per_threshold) for per_threshold in canonical],
    )


def _collapse_missing_cuts(
    *,
    cuts: np.ndarray,
    crossed: np.ndarray,
    n_thresholds: int,
    n_crossings: Sequence[int],
    directions: Sequence[int],
    starts_above: np.ndarray,
    has_valid: np.ndarray,
    valid_start: np.ndarray,
    valid_end: np.ndarray,
) -> None:
    """Park every branch a row did not cross on an edge of its valid interval.

    A branch missing from a row is a branch that ran off the side of the valid
    area, so its band there has no width.  Parking it past the last surviving
    branch, or before the first, keeps the band count the same in every row
    while giving the empty band zero width.

    ``cuts`` is filled in place.

    Parameters
    ----------
    cuts : numpy.ndarray
        Cut positions, shape ``(n_cuts, n_rows)``, ``NaN`` where uncrossed.
    crossed : numpy.ndarray
        Whether each cut is a real crossing.
    n_thresholds : int
        Number of thresholds.
    n_crossings : sequence of int
        Branches per threshold.
    directions : sequence of int
        Direction of each branch, indexed by :func:`_cut_index`.
    starts_above : numpy.ndarray
        Whether each row starts at or above each threshold.
    has_valid : numpy.ndarray
        Rows with at least one valid point.
    valid_start, valid_end : numpy.ndarray
        Edges of each row's valid interval.
    """
    for k in range(n_thresholds):
        indices = [
            _cut_index(n_crossings=n_crossings, threshold=k, crossing=b)
            for b in range(n_crossings[k])
        ]
        for row in np.flatnonzero(has_valid):
            missing = [i for i in indices if not crossed[i, row]]
            if not missing:
                continue
            present = [i for i in indices if crossed[i, row]]
            if present:
                # Branches before the first survivor collapse onto the low edge
                # and those after the last onto the high edge; any in between
                # sit on their neighbour, which is already the right answer.
                for i in missing:
                    if i < present[0]:
                        cuts[i, row] = valid_start[row]
                    elif i > present[-1]:
                        cuts[i, row] = valid_end[row]
                    else:
                        cuts[i, row] = cuts[max(p for p in present if p < i), row]
            elif n_crossings[k] == 1:
                # The single branch lies past whichever edge the output would
                # have reached it beyond.
                cuts[indices[0], row] = (
                    valid_start[row]
                    if starts_above[k, row] == (directions[indices[0]] > 0)
                    else valid_end[row]
                )
            else:
                # The threshold is never met in this row, so every band it
                # bounds is empty; stack them all on the low edge.
                for i in missing:
                    cuts[i, row] = valid_start[row]


def _lay_out_bands(
    *,
    cuts: np.ndarray,
    crossed: np.ndarray,
    n_thresholds: int,
    n_crossings: Sequence[int],
    directions: Sequence[int],
    starts_above: np.ndarray,
    has_valid: np.ndarray,
    valid_start: np.ndarray,
) -> tuple[np.ndarray, list[int]] | None:
    """Order each row's cuts and work out which region every band belongs to.

    Walks each row from the low end of the scan axis, stepping the region index
    up at a rising crossing and down at a falling one.  Every row has to
    produce the same sequence, or a band would change colour partway up the
    chart.

    Parameters
    ----------
    cuts : numpy.ndarray
        Refined cut positions, shape ``(n_cuts, n_rows)``.
    crossed : numpy.ndarray
        Whether each cut is a real crossing rather than one collapsed onto the
        edge of the valid area.
    n_thresholds : int
        Number of thresholds.
    n_crossings : sequence of int
        Crossings per threshold.
    directions : sequence of int
        Direction of each cut, indexed by :func:`_cut_index`.
    starts_above : numpy.ndarray
        Whether each row starts at or above each threshold.
    has_valid : numpy.ndarray
        Rows with at least one valid point.
    valid_start : numpy.ndarray
        Low edge of each row's valid interval.

    Returns
    -------
    tuple or None
        ``(order, band_regions)``, where ``order`` gives the cut indices sorted
        along the scan axis per row; ``None`` if the rows disagree.
    """
    n_cuts, n_rows = cuts.shape
    owner = [k for k in range(n_thresholds) for _ in range(n_crossings[k])]
    order = np.zeros((n_cuts, n_rows), dtype=int)
    reference: tuple | None = None
    reference_region: int | None = None

    for row in range(n_rows):
        if not has_valid[row]:
            continue

        # Ties happen where uncrossed thresholds collapse onto the same edge.
        # A rising output meets the lower threshold first and a falling one the
        # higher, so break ties by threshold index signed with the direction.
        row_order = sorted(
            range(n_cuts),
            key=lambda i: (cuts[i, row], directions[i] * owner[i]),
        )
        order[:, row] = row_order

        region = int(starts_above[:, row].sum())
        # A cut sitting on the low edge has already been stepped over by the
        # time the row's first sample is read, so undo it.
        for i in row_order:
            if not crossed[i, row] and cuts[i, row] <= valid_start[row]:
                region -= directions[i]

        signature = tuple((owner[i], directions[i]) for i in row_order)
        if reference is None:
            reference, reference_region = signature, region
        elif signature != reference or region != reference_region:
            return None

    if reference_region is None:
        return None

    first_valid = int(np.flatnonzero(has_valid)[0])
    band_regions = [reference_region]
    for i in order[:, first_valid]:
        band_regions.append(band_regions[-1] + directions[i])

    return order, band_regions


def solve_region_bands(
    *,
    evaluate: EvaluateFn,
    scan: np.ndarray,
    rows: np.ndarray,
    thresholds: Sequence[float],
    scan_axis: str,
) -> RegionBands | None:
    """Solve exact region boundaries by scanning one axis and bisecting.

    Parameters
    ----------
    evaluate : callable
        Takes flat ``scan`` and ``row`` coordinate arrays of equal length and
        returns the model output for each pair.
    scan : numpy.ndarray
        Sample coordinates along the scanned axis, ascending.  These only need
        to be fine enough to bracket each crossing separately; the bisection
        supplies the precision.
    rows : numpy.ndarray
        Coordinates along the perpendicular axis, one per row.  These set how
        finely the boundary curves are sampled.
    thresholds : sequence of float
        Threshold values, ascending.
    scan_axis : {"x", "y"}
        Which plot axis ``scan`` refers to.

    Returns
    -------
    RegionBands or None
        Solved bands, or ``None`` when the geometry cannot be laid out the
        same way in every row.
    """
    scan = np.asarray(scan, dtype=float)
    rows = np.asarray(rows, dtype=float)
    n_rows, n_scan = rows.size, scan.size
    n_thresholds = len(thresholds)

    scan_mesh, row_mesh = np.meshgrid(scan, rows)
    grid = np.asarray(evaluate(scan_mesh, row_mesh), dtype=float).reshape(
        n_rows, n_scan
    )

    scanned = _scan_rows(grid=grid, scan=scan, thresholds=thresholds)
    if scanned is None:
        return None

    valid_start = scanned.valid_start
    valid_end = scanned.valid_end
    has_valid = scanned.has_valid
    n_crossings = scanned.n_crossings
    directions = scanned.directions
    n_cuts = sum(n_crossings)

    cuts = np.full((n_cuts, n_rows), np.nan)
    crossed = np.zeros((n_cuts, n_rows), dtype=bool)

    if scanned.brackets:
        brackets = scanned.brackets
        roots = _bisect(
            evaluate=evaluate,
            lo=np.array([b[0] for b in brackets], dtype=float),
            hi=np.array([b[1] for b in brackets], dtype=float),
            rows=rows[np.array([b[2] for b in brackets], dtype=int)],
            targets=np.array([b[3] for b in brackets], dtype=float),
            predicate_at_lo=np.array([b[4] for b in brackets], dtype=bool),
            tol=abs(scan[-1] - scan[0]) * _REL_TOL,
        )
        if roots is None:
            return None

        for root, bracket in zip(roots, brackets, strict=True):
            slot = bracket.slot
            if isinstance(slot, _ValidityEdge):
                edge = valid_end if slot.high_end else valid_start
                edge[slot.row] = root
            else:
                index = _cut_index(
                    n_crossings=n_crossings,
                    threshold=slot.threshold,
                    crossing=slot.branch,
                )
                cuts[index, slot.row] = root
                crossed[index, slot.row] = True

    _collapse_missing_cuts(
        cuts=cuts,
        crossed=crossed,
        n_thresholds=n_thresholds,
        n_crossings=n_crossings,
        directions=directions,
        starts_above=scanned.starts_above,
        has_valid=has_valid,
        valid_start=valid_start,
        valid_end=valid_end,
    )

    laid_out = _lay_out_bands(
        cuts=cuts,
        crossed=crossed,
        n_thresholds=n_thresholds,
        n_crossings=n_crossings,
        directions=directions,
        starts_above=scanned.starts_above,
        has_valid=has_valid,
        valid_start=valid_start,
    )
    if laid_out is None:
        return None
    order, band_regions = laid_out

    curves: list[BoundaryCurve] = []
    for k, threshold in enumerate(thresholds):
        for crossing in range(n_crossings[k]):
            index = _cut_index(n_crossings=n_crossings, threshold=k, crossing=crossing)
            # NaN wherever the threshold is not actually crossed, so the curve
            # breaks instead of being dragged to the edge of the valid area.
            curve_x, curve_y = _to_xy(
                scan_axis=scan_axis,
                scan_coords=np.where(crossed[index], cuts[index], np.nan),
                rows=rows,
            )
            curves.append(
                BoundaryCurve(
                    threshold=float(threshold),
                    branch=crossing,
                    x=curve_x,
                    y=curve_y,
                )
            )

    has_invalid = bool(
        (~has_valid).any()
        or np.any(valid_start[has_valid] > scan[0])
        or np.any(valid_end[has_valid] < scan[-1])
    )

    # Rows with no valid point collapse to the low end of the scan axis, so the
    # out-of-limits shading spans the whole row and every band vanishes.
    valid_start = np.where(has_valid, valid_start, scan[0])
    valid_end = np.where(has_valid, valid_end, scan[0])

    ordered = np.take_along_axis(cuts, order, axis=0)
    edges = np.vstack([valid_start[None, :], ordered, valid_end[None, :]])
    edges = np.where(has_valid[None, :], edges, scan[0])
    # Guard against float noise reordering two boundaries that nearly coincide.
    edges = np.maximum.accumulate(edges, axis=0)
    edges = np.minimum(edges, valid_end[None, :])

    # Stacking the branches a row never crossed can walk the region index past
    # either end of the region list.  That is only meaningful if such a band
    # ever has width -- otherwise it is an empty sliver and the caller drops it.
    width = np.abs(np.diff(edges, axis=0)).max(axis=1)
    for band, region in enumerate(band_regions):
        if not 0 <= region <= n_thresholds and width[band] > 0:
            return None

    return RegionBands(
        scan_axis=scan_axis,
        rows=rows,
        edges=edges,
        band_regions=band_regions,
        valid_start=valid_start,
        valid_end=valid_end,
        has_valid=has_valid,
        has_invalid=has_invalid,
        curves=curves,
    )


def _to_xy(
    scan_axis: str, scan_coords: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map scan/row coordinates to ``(x, y)`` for the given scan axis."""
    if scan_axis == "x":
        return scan_coords, rows
    return rows, scan_coords
