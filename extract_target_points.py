import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractionResult:
    target_points: np.ndarray  # shape: (N, 2) in (x, y)
    binary: np.ndarray  # uint8 {0,255}
    skeleton: np.ndarray | None = None  # optional uint8 {0,255} skeleton overlay (same shape as binary)


def _ensure_mpl_configdir_writable() -> None:
    """
    Avoid Matplotlib cache warnings on machines where ~/.matplotlib is not writable.
    We only set this if it's missing; it won't affect OpenCV/Numpy logic.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def load_and_binarize(image_path: str, threshold: int = 127) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return image, binary


def _sort_contours_left_to_right(contours: list[np.ndarray]) -> list[np.ndarray]:
    def key_fn(c: np.ndarray) -> int:
        x, y, w, h = cv2.boundingRect(c)
        return x

    return sorted(contours, key=key_fn)


def _sample_points_along_closed_polyline(points_xy: np.ndarray, n: int) -> np.ndarray:
    """
    points_xy: (M, 2) polyline points in (x, y) order (contour order).
    We treat the contour as closed and sample n points at equal arc-length spacing.
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if len(points_xy) == 1:
        return np.repeat(points_xy.astype(np.int32), repeats=n, axis=0)

    pts = points_xy.astype(np.float64)
    pts_closed = np.vstack([pts, pts[:1]])
    diffs = np.diff(pts_closed, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cum[-1])

    if total <= 1e-9:
        return np.repeat(points_xy[:1].astype(np.int32), repeats=n, axis=0)

    desired = np.linspace(0.0, total, n, endpoint=False)
    idx = np.searchsorted(cum, desired, side="right") - 1
    idx = np.clip(idx, 0, len(points_xy) - 1)
    return points_xy[idx].astype(np.int32)


def _sample_points_inside_mask(
    binary: np.ndarray,
    n: int,
    *,
    min_border_dist: float = 3.0,
    topk_candidates: int = 5000,
    seed: int | None = None,
    fixed_points: np.ndarray | None = None,  
) -> np.ndarray:
    """
    Sample points from the *interior* (white) region of `binary` (255=foreground),
    preferring pixels far from the border and spreading points out.
    Uses distance transform + greedy maximin selection on top-K candidates.
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    mask = binary > 0

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dvals = dist[ys, xs]

    keep = dvals >= float(min_border_dist)
    if np.any(keep):
        ys, xs, dvals = ys[keep], xs[keep], dvals[keep]

    if len(xs) < n:
        ys, xs = np.nonzero(mask)
        dvals = dist[ys, xs]

    k = int(min(int(topk_candidates), len(xs)))
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    if k < len(xs):
        idx = np.argpartition(dvals, -k)[-k:]
        ys, xs, dvals = ys[idx], xs[idx], dvals[idx]

    
    coords = np.column_stack([xs, ys]).astype(np.float64) 
    dvals = dvals.astype(np.float64)

    min_d2 = np.full(len(coords), np.inf, dtype=np.float64)
    selected: list[int] = []

    rng = np.random.default_rng(seed) if seed is not None else None

    fixed = np.zeros((0, 2), dtype=np.int32)
    if fixed_points is not None and len(fixed_points) > 0:
        fixed = np.asarray(fixed_points, dtype=np.int32).reshape(-1, 2)

    if len(fixed) > 0:
        for (fx, fy) in fixed:
            dx = coords[:, 0] - float(fx)
            dy = coords[:, 1] - float(fy)
            min_d2 = np.minimum(min_d2, dx * dx + dy * dy)
            same = np.where((coords[:, 0] == float(fx)) & (coords[:, 1] == float(fy)))[0]
            if len(same) > 0:
                selected.append(int(same[0]))
    else:
        first = int(np.argmax(dvals))
        selected = [first]
        dx = coords[:, 0] - coords[first, 0]
        dy = coords[:, 1] - coords[first, 1]
        min_d2 = np.minimum(min_d2, dx * dx + dy * dy)

    for _ in range(len(selected), n):
        min_d = np.sqrt(min_d2)
        score = dvals * (min_d + 1e-6)
        score[selected] = -1.0  # never re-pick

        best = int(np.argmax(score))
        if score[best] <= 0:
            remaining = np.where(score >= 0)[0]
            if len(remaining) == 0:
                break
            best = int(rng.choice(remaining)) if rng is not None else int(remaining[0])

        selected.append(best)

        dx = coords[:, 0] - coords[best, 0]
        dy = coords[:, 1] - coords[best, 1]
        min_d2 = np.minimum(min_d2, dx * dx + dy * dy)

    pts = coords[np.array(selected, dtype=int)].round().astype(np.int32)
    if len(fixed) > 0:
        combined = np.vstack([fixed, pts]) if len(pts) else fixed.copy()
        seen: set[tuple[int, int]] = set()
        uniq = []
        for x, y in combined.tolist():
            t = (int(x), int(y))
            if t in seen:
                continue
            if not mask[t[1], t[0]]:
                continue
            uniq.append(t)
            seen.add(t)
            if len(uniq) >= n:
                break
        pts = np.array(uniq, dtype=np.int32) if uniq else np.zeros((0, 2), dtype=np.int32)
    if len(pts) < n and len(pts) > 0:
        reps = n - len(pts)
        pts = np.vstack([pts, pts[:reps]])
    return pts


def _zhang_suen_thinning(fg_mask: np.ndarray) -> np.ndarray:
    """
    Zhang–Suen thinning (skeletonization) for a binary foreground mask.
    Input: fg_mask bool/uint8, where True/1 means foreground.
    Output: bool mask of the skeleton.
    """
    img = (fg_mask > 0).astype(np.uint8)
    if img.ndim != 2:
        raise ValueError("thinning expects a 2D mask")
    if img.sum() == 0:
        return img.astype(bool)

    changed = True
    while changed:
        changed = False

        P1 = img[1:-1, 1:-1]
        P2 = img[:-2, 1:-1]
        P3 = img[:-2, 2:]
        P4 = img[1:-1, 2:]
        P5 = img[2:, 2:]
        P6 = img[2:, 1:-1]
        P7 = img[2:, :-2]
        P8 = img[1:-1, :-2]
        P9 = img[:-2, :-2]

        N = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        S = (
            ((P2 == 0) & (P3 == 1)).astype(np.uint8)
            + ((P3 == 0) & (P4 == 1)).astype(np.uint8)
            + ((P4 == 0) & (P5 == 1)).astype(np.uint8)
            + ((P5 == 0) & (P6 == 1)).astype(np.uint8)
            + ((P6 == 0) & (P7 == 1)).astype(np.uint8)
            + ((P7 == 0) & (P8 == 1)).astype(np.uint8)
            + ((P8 == 0) & (P9 == 1)).astype(np.uint8)
            + ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        )

        cond1 = (P1 == 1) & (N >= 2) & (N <= 6) & (S == 1)
        cond2 = (P2 * P4 * P6) == 0
        cond3 = (P4 * P6 * P8) == 0
        m = cond1 & cond2 & cond3
        if np.any(m):
            img[1:-1, 1:-1][m] = 0
            changed = True

        P1 = img[1:-1, 1:-1]
        P2 = img[:-2, 1:-1]
        P3 = img[:-2, 2:]
        P4 = img[1:-1, 2:]
        P5 = img[2:, 2:]
        P6 = img[2:, 1:-1]
        P7 = img[2:, :-2]
        P8 = img[1:-1, :-2]
        P9 = img[:-2, :-2]

        N = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        S = (
            ((P2 == 0) & (P3 == 1)).astype(np.uint8)
            + ((P3 == 0) & (P4 == 1)).astype(np.uint8)
            + ((P4 == 0) & (P5 == 1)).astype(np.uint8)
            + ((P5 == 0) & (P6 == 1)).astype(np.uint8)
            + ((P6 == 0) & (P7 == 1)).astype(np.uint8)
            + ((P7 == 0) & (P8 == 1)).astype(np.uint8)
            + ((P8 == 0) & (P9 == 1)).astype(np.uint8)
            + ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        )

        cond1 = (P1 == 1) & (N >= 2) & (N <= 6) & (S == 1)
        cond2 = (P2 * P4 * P8) == 0
        cond3 = (P2 * P6 * P8) == 0
        m = cond1 & cond2 & cond3
        if np.any(m):
            img[1:-1, 1:-1][m] = 0
            changed = True

    return img.astype(bool)


def _sample_points_from_coords(
    coords_xy: np.ndarray,
    n: int,
    *,
    fixed_points: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Greedy maximin sampling from a discrete set of (x,y) coordinates.
    Ensures points lie exactly on the provided locus (e.g. skeleton pixels).
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    if coords_xy is None or len(coords_xy) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    coords = np.asarray(coords_xy, dtype=np.float64).reshape(-1, 2)
    rng = np.random.default_rng(seed) if seed is not None else None

    fixed: list[tuple[int, int]] = []
    if fixed_points is not None and len(fixed_points) > 0:
        for x, y in np.asarray(fixed_points, dtype=np.int32).reshape(-1, 2).tolist():
            fixed.append((int(x), int(y)))
        fixed = list(dict.fromkeys(fixed))  # preserve order, unique

    selected_idx: list[int] = []
    if fixed:
        for fx, fy in fixed:
            m = np.where((coords[:, 0] == float(fx)) & (coords[:, 1] == float(fy)))[0]
            if len(m) > 0:
                selected_idx.append(int(m[0]))

    if not selected_idx:
        c = coords.mean(axis=0)
        d2 = (coords[:, 0] - c[0]) ** 2 + (coords[:, 1] - c[1]) ** 2
        first = int(np.argmax(d2))
        selected_idx = [first]

    min_d2 = np.full(len(coords), np.inf, dtype=np.float64)
    for si in selected_idx:
        dx = coords[:, 0] - coords[si, 0]
        dy = coords[:, 1] - coords[si, 1]
        min_d2 = np.minimum(min_d2, dx * dx + dy * dy)

    while len(selected_idx) < n:
        score = min_d2.copy()
        score[selected_idx] = -1.0
        best = int(np.argmax(score))
        if score[best] <= 0:
            remaining = np.where(score >= 0)[0]
            if len(remaining) == 0:
                break
            best = int(rng.choice(remaining)) if rng is not None else int(remaining[0])
        selected_idx.append(best)
        dx = coords[:, 0] - coords[best, 0]
        dy = coords[:, 1] - coords[best, 1]
        min_d2 = np.minimum(min_d2, dx * dx + dy * dy)

    pts = coords[np.array(selected_idx, dtype=int)].round().astype(np.int32)

    if fixed:
        combined = np.vstack([np.array(fixed, dtype=np.int32), pts])
        seen: set[tuple[int, int]] = set()
        uniq: list[tuple[int, int]] = []
        for x, y in combined.tolist():
            t = (int(x), int(y))
            if t in seen:
                continue
            uniq.append(t)
            seen.add(t)
            if len(uniq) >= n:
                break
        pts = np.array(uniq, dtype=np.int32)

    if len(pts) < n and len(pts) > 0:
        reps = n - len(pts)
        pts = np.vstack([pts, pts[:reps]])

    return pts[:n]


def extract_target_points_contour(image_path: str, n_drones: int = 50, threshold: int = 127) -> ExtractionResult:
    _, binary = load_and_binarize(image_path, threshold=threshold)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours found. Try adjusting threshold or check the image.")

    contours = _sort_contours_left_to_right(contours)

    perimeters = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float64)
    total_perim = float(perimeters.sum())
    if total_perim <= 1e-9:
        raise RuntimeError("Contours are degenerate (total perimeter is zero).")

    raw_alloc = (perimeters / total_perim) * float(n_drones)
    alloc = np.floor(raw_alloc).astype(int)
    if n_drones >= len(contours):
        alloc = np.maximum(alloc, 1)

    diff = int(n_drones - int(alloc.sum()))
    if diff > 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(-frac)
        for i in range(diff):
            alloc[order[i % len(contours)]] += 1
    elif diff < 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(frac)  # smallest first
        min_per_contour = 1 if n_drones >= len(contours) else 0
        i = 0
        while diff < 0 and i < 10_000:
            ci = int(order[i % len(contours)])
            if alloc[ci] > min_per_contour:
                alloc[ci] -= 1
                diff += 1
            i += 1

    sampled = []
    for c, k in zip(contours, alloc.tolist()):
        if k <= 0:
            continue
        pts = c.reshape(-1, 2)  # (x, y)
        sampled.append(_sample_points_along_closed_polyline(pts, k))

    target_points = np.vstack(sampled) if sampled else np.zeros((0, 2), dtype=np.int32)

    if len(target_points) > n_drones:
        target_points = target_points[:n_drones]
    elif len(target_points) < n_drones and len(target_points) > 0:
        reps = n_drones - len(target_points)
        target_points = np.vstack([target_points, target_points[:reps]])

    return ExtractionResult(target_points=target_points.astype(np.int32), binary=binary, skeleton=None)


def extract_target_points_interior(
    image_path: str,
    n_drones: int = 500,
    threshold: int = 127,
    *,
    min_border_dist: float = 3.0,
    topk_candidates: int = 5000,
    seed: int | None = None,
    min_per_component: int = 8,
    use_extremes: bool = True,
    extreme_directions: int = 8,
) -> ExtractionResult:
    _, binary = load_and_binarize(image_path, threshold=threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    comp_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0]
    if not comp_ids:
        return ExtractionResult(target_points=np.zeros((0, 2), dtype=np.int32), binary=binary, skeleton=None)

    comp_ids.sort(key=lambda i: float(centroids[i][0]))

    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in comp_ids], dtype=np.float64)
    total_area = float(areas.sum())
    raw_alloc = (areas / total_area) * float(n_drones)
    alloc = np.floor(raw_alloc).astype(int)

    if len(comp_ids) > 0:
        max_min = max(1, n_drones // len(comp_ids))
        min_pc = int(min(max_min, max(1, min_per_component)))
        alloc = np.maximum(alloc, min_pc)

    diff = int(n_drones - int(alloc.sum()))
    if diff > 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(-frac)
        for i in range(diff):
            alloc[order[i % len(comp_ids)]] += 1
    elif diff < 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(frac)
        min_pc = int(min(alloc.min(initial=0), max(0, n_drones // len(comp_ids))))
        i = 0
        while diff < 0 and i < 100_000:
            ci = int(order[i % len(comp_ids)])
            if alloc[ci] > min_pc:
                alloc[ci] -= 1
                diff += 1
            i += 1

    def inward_from_contour(
        component_mask: np.ndarray, dist_map: np.ndarray, contour_xy: np.ndarray, centroid_xy: tuple[float, float]
    ) -> np.ndarray:
        if len(contour_xy) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        cx, cy = centroid_xy
        dirs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        if extreme_directions >= 8:
            d = 1.0 / np.sqrt(2.0)
            dirs += [(d, d), (d, -d), (-d, d), (-d, -d)]

        pts = contour_xy.astype(np.float64)
        seeds: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        h, w = component_mask.shape[:2]
        for dx, dy in dirs:
            proj = pts[:, 0] * dx + pts[:, 1] * dy
            p = pts[int(np.argmax(proj))]
            x0, y0 = float(p[0]), float(p[1])

            vx, vy = (cx - x0), (cy - y0)
            norm = float(np.hypot(vx, vy))
            if norm <= 1e-9:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = vx / norm, vy / norm

            best: tuple[int, int] | None = None
            for t in range(0, int(max(w, h))):
                xi = int(round(x0 + vx * t))
                yi = int(round(y0 + vy * t))
                if xi < 0 or yi < 0 or xi >= w or yi >= h:
                    break
                if component_mask[yi, xi] == 0:
                    continue
                best = (xi, yi)
                if dist_map[yi, xi] >= float(min_border_dist):
                    break
            if best is None:
                continue
            if best not in seen:
                seeds.append(best)
                seen.add(best)

        return np.array(seeds, dtype=np.int32)

    all_pts: list[np.ndarray] = []
    for cid, k in zip(comp_ids, alloc.tolist()):
        component_mask = ((labels == cid).astype(np.uint8) * 255)
        if k <= 0:
            continue

        fixed = None
        if use_extremes:
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = max(contours, key=lambda cc: len(cc))
                contour_xy = c.reshape(-1, 2)
                dist_map = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
                cx, cy = float(centroids[cid][0]), float(centroids[cid][1])
                fixed = inward_from_contour(component_mask, dist_map, contour_xy, (cx, cy))
                if fixed is not None and len(fixed) > max(0, k - 1):
                    fixed = fixed[: max(0, k - 1)]

        pts_c = _sample_points_inside_mask(
            component_mask,
            k,
            min_border_dist=min_border_dist,
            topk_candidates=topk_candidates,
            seed=seed,
            fixed_points=fixed,
        )
        all_pts.append(pts_c)

    pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2), dtype=np.int32)
    if len(pts) > n_drones:
        pts = pts[:n_drones]
    elif len(pts) < n_drones and len(pts) > 0:
        reps = n_drones - len(pts)
        pts = np.vstack([pts, pts[:reps]])
    return ExtractionResult(target_points=pts.astype(np.int32), binary=binary, skeleton=None)


def extract_target_points_skeleton(
    image_path: str,
    n_drones: int = 500,
    threshold: int = 127,
    *,
    min_per_component: int = 8,
    seed: int | None = None,
) -> ExtractionResult:
    """
    Place target points *exactly* on the medial axis (skeleton) of each connected component.
    """
    _, binary = load_and_binarize(image_path, threshold=threshold)
    fg = (binary > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
    comp_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0]
    if not comp_ids:
        return ExtractionResult(target_points=np.zeros((0, 2), dtype=np.int32), binary=binary, skeleton=np.zeros_like(binary))

    comp_ids.sort(key=lambda i: float(centroids[i][0]))

    skeleton_overlay = np.zeros_like(binary)
    skel_counts = []
    skel_coords_per_comp: list[np.ndarray] = []
    endpoints_per_comp: list[np.ndarray] = []

    for cid in comp_ids:
        component = (labels == cid)
        skel = _zhang_suen_thinning(component.astype(np.uint8))
        skeleton_overlay[skel] = 255

        ys, xs = np.nonzero(skel)
        coords = np.column_stack([xs, ys]).astype(np.int32)
        skel_coords_per_comp.append(coords)
        skel_counts.append(len(coords))

        if len(coords) == 0:
            endpoints_per_comp.append(np.zeros((0, 2), dtype=np.int32))
            continue
        sk = skel.astype(np.uint8)
        ncount = (
            sk[:-2, 1:-1]
            + sk[:-2, 2:]
            + sk[1:-1, 2:]
            + sk[2:, 2:]
            + sk[2:, 1:-1]
            + sk[2:, :-2]
            + sk[1:-1, :-2]
            + sk[:-2, :-2]
        )
        core = sk[1:-1, 1:-1].astype(bool)
        ep = core & (ncount == 1)
        epy, epx = np.nonzero(ep)
        ep_coords = np.column_stack([epx + 1, epy + 1]).astype(np.int32)
        endpoints_per_comp.append(ep_coords)

    skel_counts_arr = np.array(skel_counts, dtype=np.float64)
    total = float(skel_counts_arr.sum())
    if total <= 0:
        res = extract_target_points_interior(image_path, n_drones=n_drones, threshold=threshold, seed=seed)
        return ExtractionResult(target_points=res.target_points, binary=binary, skeleton=skeleton_overlay)

    raw_alloc = (skel_counts_arr / total) * float(n_drones)
    alloc = np.floor(raw_alloc).astype(int)

    if len(comp_ids) > 0:
        max_min = max(1, n_drones // len(comp_ids))
        min_pc = int(min(max_min, max(1, min_per_component)))
        alloc = np.maximum(alloc, min_pc)

    diff = int(n_drones - int(alloc.sum()))
    if diff > 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(-frac)
        for i in range(diff):
            alloc[order[i % len(comp_ids)]] += 1
    elif diff < 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(frac)
        min_pc = int(min(alloc.min(initial=0), max(0, n_drones // len(comp_ids))))
        i = 0
        while diff < 0 and i < 100_000:
            ci = int(order[i % len(comp_ids)])
            if alloc[ci] > min_pc:
                alloc[ci] -= 1
                diff += 1
            i += 1

    all_pts: list[np.ndarray] = []
    for coords, eps, k in zip(skel_coords_per_comp, endpoints_per_comp, alloc.tolist()):
        if k <= 0 or len(coords) == 0:
            continue
        fixed = eps
        if fixed is not None and len(fixed) > max(0, k - 1):
            fixed = fixed[: max(0, k - 1)]
        pts_c = _sample_points_from_coords(coords, k, fixed_points=fixed, seed=seed)
        all_pts.append(pts_c)

    pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2), dtype=np.int32)
    if len(pts) > n_drones:
        pts = pts[:n_drones]
    elif len(pts) < n_drones and len(pts) > 0:
        reps = n_drones - len(pts)
        pts = np.vstack([pts, pts[:reps]])

    return ExtractionResult(target_points=pts.astype(np.int32), binary=binary, skeleton=skeleton_overlay)


def save_outputs(
    out_dir: str,
    target_points: np.ndarray,
    binary: np.ndarray,
    n_drones: int,
    save_debug_png: bool,
    debug_point_radius: int,
    skeleton: np.ndarray | None = None,
) -> tuple[str, str, str | None]:
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "target_points.npy")
    csv_path = os.path.join(out_dir, "target_points.csv")
    debug_path = os.path.join(out_dir, "debug_target_points.png") if save_debug_png else None

    np.save(npy_path, target_points)
    np.savetxt(csv_path, target_points, fmt="%d", delimiter=",", header="x,y", comments="")

    if save_debug_png:
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if skeleton is not None:
            sk = (skeleton > 0)
            vis[sk] = (0, 180, 0)
        for (x, y) in target_points:
            cv2.circle(vis, (int(x), int(y)), int(debug_point_radius), (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"N={n_drones}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(debug_path, vis)

    return npy_path, csv_path, debug_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract equally-spaced target points along text contours.")
    parser.add_argument("--image", default="name.png", help="Path to input image (e.g. name.png)")
    parser.add_argument("--n", type=int, default=500, help="Number of target points (drones)")
    parser.add_argument("--threshold", type=int, default=127, help="Binary threshold (0..255)")
    parser.add_argument("--out-dir", default=".", help="Output directory for target_points.* files")
    parser.add_argument(
        "--mode",
        choices=["interior", "contour", "skeleton"],
        default="interior",
        help="Point placement mode: interior (inside white region), contour (on outer edges), or skeleton (medial axis).",
    )
    parser.add_argument(
        "--min-border-dist",
        type=float,
        default=3.0,
        help="(interior mode) Minimum distance from border in pixels (higher -> more centered).",
    )
    parser.add_argument(
        "--topk-candidates",
        type=int,
        default=5000,
        help="(interior mode) Candidate pixels considered (top-K by distance-to-border).",
    )
    parser.add_argument("--seed", type=int, default=None, help="(interior mode) RNG seed for tie-breaking")
    parser.add_argument(
        "--min-per-component",
        type=int,
        default=8,
        help="(interior/skeleton mode) Minimum points per connected component (letter).",
    )
    parser.add_argument(
        "--no-extremes",
        action="store_true",
        help="(interior mode) Disable forcing a few points toward letter extremities (tips).",
    )
    parser.add_argument(
        "--extreme-dirs",
        type=int,
        default=8,
        choices=[4, 8],
        help="(interior mode) Number of extreme directions to seed per letter (4 or 8).",
    )
    parser.add_argument("--debug-png", action="store_true", help="Also save debug_target_points.png")
    parser.add_argument("--debug-point-radius", type=int, default=3, help="Point radius for debug PNG (pixels)")
    parser.add_argument("--show", action="store_true", help="Show matplotlib visualization window")
    parser.add_argument("--show-point-size", type=int, default=70, help="Point size for --show (matplotlib scatter)")
    args = parser.parse_args()

    if args.mode == "contour":
        res = extract_target_points_contour(args.image, n_drones=args.n, threshold=args.threshold)
    elif args.mode == "skeleton":
        res = extract_target_points_skeleton(
            args.image,
            n_drones=args.n,
            threshold=args.threshold,
            min_per_component=args.min_per_component,
            seed=args.seed,
        )
    else:
        res = extract_target_points_interior(
            args.image,
            n_drones=args.n,
            threshold=args.threshold,
            min_border_dist=args.min_border_dist,
            topk_candidates=args.topk_candidates,
            seed=args.seed,
            min_per_component=args.min_per_component,
            use_extremes=not args.no_extremes,
            extreme_directions=args.extreme_dirs,
        )
    npy_path, csv_path, debug_path = save_outputs(
        out_dir=args.out_dir,
        target_points=res.target_points,
        binary=res.binary,
        n_drones=args.n,
        save_debug_png=args.debug_png,
        debug_point_radius=args.debug_point_radius,
        skeleton=res.skeleton,
    )

    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")
    if debug_path:
        print(f"Saved: {debug_path}")
    print(f"target_points shape: {res.target_points.shape}")
    print("first 5 points (x,y):")
    print(res.target_points[:5])

    if args.show:
        _ensure_mpl_configdir_writable()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.imshow(res.binary, cmap="gray")
        plt.scatter(res.target_points[:, 0], res.target_points[:, 1], c="red", s=args.show_point_size)
        plt.title(f"Contours + {args.n} sampled points")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

