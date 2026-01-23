import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np


def _ensure_mpl_configdir_writable() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def load_points(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        pts = np.load(path)
    else:
        pts = np.loadtxt(path, delimiter=",", skiprows=1)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points in {path}, got {pts.shape}")
    return pts


def _mask_to_largest_filled(mask_u8: np.ndarray, *, ignore_bottom_frac: float) -> np.ndarray:
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    if ignore_bottom_frac > 0:
        h, w = mask.shape[:2]
        cut = int(round((1.0 - float(ignore_bottom_frac)) * h))
        cut = int(np.clip(cut, 0, h))
        if cut < h:
            mask[cut:, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask)
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out


def segment_greenscreen(frame_bgr: np.ndarray, *, ignore_bottom_frac: float = 0.08) -> np.ndarray:
    """
    Segment a dark silhouette on a green background.
    Returns mask uint8 {0,255}.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 60), (90, 255, 255))
    mask = cv2.bitwise_not(green)
    return _mask_to_largest_filled(mask, ignore_bottom_frac=float(ignore_bottom_frac))


def segment_mog2(
    frame_bgr: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    *,
    ignore_bottom_frac: float = 0.08,
    threshold: int = 200,
) -> np.ndarray:
    fg = subtractor.apply(frame_bgr)
    fg = (fg >= int(threshold)).astype(np.uint8) * 255
    return _mask_to_largest_filled(fg, ignore_bottom_frac=float(ignore_bottom_frac))


def segment_edges(
    frame_bgr: np.ndarray,
    *,
    ignore_bottom_frac: float = 0.08,
    canny1: int = 60,
    canny2: int = 150,
    dilate_iters: int = 1,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(canny1), int(canny2))
    if int(dilate_iters) > 0:
        k = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k, iterations=int(dilate_iters))
    return _mask_to_largest_filled(edges, ignore_bottom_frac=float(ignore_bottom_frac))


def _extract_largest_contour_points(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float64)


def _smooth_closed_polyline(points_xy: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    w = int(window)
    if n == 0 or w <= 1:
        return pts
    if w % 2 == 0:
        w += 1
    r = w // 2
    out = np.zeros_like(pts)
    for i in range(n):
        idx = [(i + j) % n for j in range(-r, r + 1)]
        out[i] = pts[idx].mean(axis=0)
    return out


def _extract_contour_points_highres(mask_u8: np.ndarray, *, upscale: float, smooth_window: int) -> np.ndarray:
    s = float(upscale)
    if s <= 1.0:
        pts = _extract_largest_contour_points(mask_u8)
        return _smooth_closed_polyline(pts, smooth_window)
    h, w = mask_u8.shape[:2]
    mask_hi = cv2.resize(mask_u8, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_NEAREST)
    pts_hi = _extract_largest_contour_points(mask_hi)
    pts_hi = _smooth_closed_polyline(pts_hi, smooth_window)
    return pts_hi / s


def _polygon_signed_area(pts_xy: np.ndarray) -> float:
    pts = np.asarray(pts_xy, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - x2 * y))


def _sample_points_along_closed_polyline_interp(points_xy: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, repeats=n, axis=0)

    pts_closed = np.vstack([pts, pts[:1]])
    seg = np.diff(pts_closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1], repeats=n, axis=0)

    desired = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(desired):
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        j = min(j, len(seg_len) - 1)
        ds = s - cum[j]
        L = float(seg_len[j])
        t = 0.0 if L <= 1e-12 else ds / L
        out[i] = pts_closed[j] + t * seg[j]
    return out


def _best_circular_alignment(prev: np.ndarray, cand: np.ndarray) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    n = prev.shape[0]
    if n == 0:
        return cand

    def best_shift(arr: np.ndarray) -> tuple[np.ndarray, float]:
        best_cost = float("inf")
        best_arr = arr
        for s in range(n):
            rolled = np.roll(arr, shift=s, axis=0)
            d = prev - rolled
            cost = float(np.sum(d * d))
            if cost < best_cost:
                best_cost = cost
                best_arr = rolled
        return best_arr, best_cost

    a1, c1 = best_shift(cand)
    a2, c2 = best_shift(cand[::-1])
    return a1 if c1 <= c2 else a2


@dataclass(frozen=True)
class DynParams:
    m: float = 1.0
    k_p: float = 4.0
    k_d: float = 2.5


def _saturate(v: np.ndarray, v_max: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = float(v_max)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.ones_like(n)
    mask = n > s
    scale[mask] = s / (n[mask] + 1e-12)
    return v * scale


def _rk4_step(pos: np.ndarray, vel: np.ndarray, *, target_pos: np.ndarray, dt: float, p: DynParams, v_max: float) -> tuple[np.ndarray, np.ndarray]:
    def deriv(pp: np.ndarray, vv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xdot = _saturate(vv, v_max)
        acc = (p.k_p * (target_pos - pp) - p.k_d * vv) / p.m
        return xdot, acc

    k1x, k1v = deriv(pos, vel)
    k2x, k2v = deriv(pos + 0.5 * dt * k1x, vel + 0.5 * dt * k1v)
    k3x, k3v = deriv(pos + 0.5 * dt * k2x, vel + 0.5 * dt * k2v)
    k4x, k4v = deriv(pos + dt * k3x, vel + dt * k3v)
    pos2 = pos + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    vel2 = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return pos2, vel2


def save_trajectories_csv(path: str, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    n, tt = x.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("drone_id,time,x,y\n")
        for i in range(n):
            for k in range(tt):
                f.write(f"{i},{t[k]:.6f},{x[i, k]:.6f},{y[i, k]:.6f}\n")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(base_dir, "video.mp4")
    default_start = os.path.join(base_dir, "..", "task2", "outputs", "target_points.csv")
    default_out_dir = os.path.join(base_dir, "outputs")

    ap = argparse.ArgumentParser(description="Task 3: contour-only tracking (high precision) + visualization.")
    ap.add_argument("--video", default=default_video)
    ap.add_argument("--start", default=default_start)
    ap.add_argument("--out-dir", default=default_out_dir)

    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--video-step", type=int, default=1)
    ap.add_argument("--transition-seconds", type=float, default=4.0)
    ap.add_argument("--ignore-bottom-frac", type=float, default=0.08)
    ap.add_argument(
        "--segmenter",
        choices=["greenscreen", "mog2", "edges"],
        default="greenscreen",
        help="Segmentation method for the moving object.",
    )
    ap.add_argument("--mog2-history", type=int, default=200)
    ap.add_argument("--mog2-var-threshold", type=float, default=16.0)
    ap.add_argument("--mog2-detect-shadows", action="store_true")
    ap.add_argument("--mog2-threshold", type=int, default=200, help="Foreground threshold (0..255)")
    ap.add_argument("--canny1", type=int, default=60)
    ap.add_argument("--canny2", type=int, default=150)
    ap.add_argument("--edge-dilate-iters", type=int, default=1)

    ap.add_argument("--tracking-mode", choices=["contour"], default="contour")
    ap.add_argument("--contour-upscale", type=float, default=3.0)
    ap.add_argument("--contour-smooth", type=int, default=9)
    ap.add_argument("--controller", choices=["direct", "dynamics"], default="direct")

    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--k-p", type=float, default=4.0)
    ap.add_argument("--k-d", type=float, default=2.5)
    ap.add_argument("--v-max", type=float, default=1e9)

    ap.add_argument("--no-bg", action="store_true")
    ap.add_argument("--drone-color", default="yellow", help="Drone marker color (e.g. blue, cyan, #1f77b4)")
    ap.add_argument("--drone-size", type=float, default=18.0, help="Drone marker size (scatter)")
    ap.add_argument("--save-gif", action="store_true")
    ap.add_argument("--output-gif", default=None)
    ap.add_argument("--gif-fps", type=int, default=30)
    ap.add_argument("--save-traj-csv", action="store_true")
    ap.add_argument("--save-traj-npy", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    start = load_points(args.start)
    n = int(args.n) if args.n is not None else int(len(start))
    if len(start) != n:
        raise ValueError(f"--start has N={len(start)} points but --n={n}. Use consistent N.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(args.video_step))
    dt = float(step) / max(1e-9, fps)

    frames_bgr: list[np.ndarray] = []
    masks_u8: list[np.ndarray] = []
    subtractor = None
    if args.segmenter == "mog2":
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(args.mog2_history),
            varThreshold=float(args.mog2_var_threshold),
            detectShadows=bool(args.mog2_detect_shadows),
        )
    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, fr = cap.read()
        if not ret:
            continue
        frames_bgr.append(fr)
        if args.segmenter == "greenscreen":
            mask = segment_greenscreen(fr, ignore_bottom_frac=float(args.ignore_bottom_frac))
        elif args.segmenter == "mog2":
            mask = segment_mog2(
                fr,
                subtractor,
                ignore_bottom_frac=float(args.ignore_bottom_frac),
                threshold=int(args.mog2_threshold),
            )
        else:
            mask = segment_edges(
                fr,
                ignore_bottom_frac=float(args.ignore_bottom_frac),
                canny1=int(args.canny1),
                canny2=int(args.canny2),
                dilate_iters=int(args.edge_dilate_iters),
            )
        masks_u8.append(mask)
    cap.release()
    if not frames_bgr:
        raise RuntimeError("No frames read from video.")

    # Per-frame contour targets with stable correspondence.
    prev = None
    targets_track = []
    for mask in masks_u8:
        contour = _extract_contour_points_highres(mask, upscale=float(args.contour_upscale), smooth_window=int(args.contour_smooth))
        if len(contour) == 0:
            samp = prev.copy() if prev is not None else start.copy()
            targets_track.append(samp)
            prev = samp
            continue

        if _polygon_signed_area(contour) < 0:
            contour = contour[::-1]
        start_idx = int(np.argmin(contour[:, 1]))  # top-most point
        contour = np.roll(contour, -start_idx, axis=0)
        samp = _sample_points_along_closed_polyline_interp(contour, n)
        if prev is not None:
            samp = _best_circular_alignment(prev, samp)
        targets_track.append(samp)
        prev = samp
    targets_track = np.stack(targets_track, axis=0)  # (K,N,2)

    # Transition to first contour pose
    n_trans = max(1, int(round(float(args.transition_seconds) / max(1e-9, dt))))
    alphas = np.linspace(0.0, 1.0, n_trans, dtype=np.float64)
    start_pos = start.astype(np.float64)
    goal_pos = targets_track[0].astype(np.float64)
    targets_trans = (1.0 - alphas)[:, None, None] * start_pos[None, :, :] + alphas[:, None, None] * goal_pos[None, :, :]

    targets_all = np.concatenate([targets_trans, targets_track], axis=0)  # (T,N,2)
    T = targets_all.shape[0]

    # Drone motion
    if args.controller == "direct":
        x_hist = targets_all[:, :, 0].T
        y_hist = targets_all[:, :, 1].T
    else:
        pos = start_pos.copy()
        vel = np.zeros_like(pos)
        x_hist = np.zeros((n, T), dtype=np.float64)
        y_hist = np.zeros((n, T), dtype=np.float64)
        p = DynParams(m=float(args.m), k_p=float(args.k_p), k_d=float(args.k_d))
        v_max = float(args.v_max)
        for k in range(T):
            x_hist[:, k] = pos[:, 0]
            y_hist[:, k] = pos[:, 1]
            pos, vel = _rk4_step(pos, vel, target_pos=targets_all[k], dt=dt, p=p, v_max=v_max)

    print(f"Video: {args.video}")
    print(f"FPS: {fps:.3f}, step={step}, dt={dt:.4f}s, sampled_frames={len(frames_bgr)}")
    print(f"N drones: {n}, total steps: {T} (transition={n_trans}, tracking={len(frames_bgr)})")
    print(f"Segmenter: {args.segmenter}")

    t_eval = np.arange(T, dtype=np.float64) * float(dt)
    if args.save_traj_csv:
        out_csv = os.path.join(out_dir, "task3_trajectories.csv")
        save_trajectories_csv(out_csv, t_eval, x_hist, y_hist)
        print(f"Saved: {out_csv}")
    if args.save_traj_npy:
        out_npy = os.path.join(out_dir, "task3_trajectories.npy")
        payload = {
            "time": t_eval,
            "x": x_hist,
            "y": y_hist,
            "start_positions": start,
            "targets": targets_all,
            "params": {
                "segmenter": str(args.segmenter),
                "controller": str(args.controller),
                "video_step": int(args.video_step),
                "transition_seconds": float(args.transition_seconds),
                "contour_upscale": float(args.contour_upscale),
                "contour_smooth": int(args.contour_smooth),
            },
        }
        np.save(out_npy, payload, allow_pickle=True)
        print(f"Saved: {out_npy}")

    # Render animation
    _ensure_mpl_configdir_writable()
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    h0, w0 = frames_bgr[0].shape[:2]
    disp_w = 640
    scale = disp_w / float(w0)
    disp_h = int(round(h0 * scale))
    bg_frames = [cv2.resize(fr, (disp_w, disp_h)) for fr in frames_bgr]
    bg_frames_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in bg_frames]
    blank = np.full((disp_h, disp_w, 3), 255, dtype=np.uint8)

    def to_disp_xy(xy: np.ndarray) -> np.ndarray:
        return xy * scale

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(blank if args.no_bg else bg_frames_rgb[0], origin="upper")
    dots = ax.scatter([], [], s=float(args.drone_size), c=str(args.drone_color))
    ax.set_title("Task 3 animation (contour tracking)")
    ax.set_xlim(0, disp_w)
    ax.set_ylim(disp_h, 0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    def update(frame: int):
        k = int(np.clip(frame, 0, T - 1))
        if not args.no_bg:
            if k < n_trans:
                im.set_data(bg_frames_rgb[0])
            else:
                im.set_data(bg_frames_rgb[min(k - n_trans, len(bg_frames_rgb) - 1)])
        pts = np.column_stack([x_hist[:, k], y_hist[:, k]])
        dots.set_offsets(to_disp_xy(pts))
        return (im, dots)

    ani = FuncAnimation(fig, update, frames=T, interval=int(round(1000 * dt)), blit=True)
    if args.save_gif:
        out_gif = os.path.join(out_dir, "task3_contour.gif") if not args.output_gif else os.path.abspath(str(args.output_gif))
        ani.save(out_gif, writer="pillow", fps=int(args.gif_fps))
        print(f"Saved: {out_gif}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

