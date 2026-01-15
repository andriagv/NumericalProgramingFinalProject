import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Weights:
    w_vel: float = 1.0   # smoothness on velocity (P[k+1]-P[k])
    w_acc: float = 0.2   # smoothness on acceleration (second difference)
    w_goal: float = 50.0  # final target matching
    w_col: float = 50.0  # collision penalty


def _ensure_mpl_configdir_writable() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def load_target_points(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        pts = np.load(path)
    else:
        pts = np.loadtxt(path, delimiter=",", skiprows=1)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"target_points must be shape (N,2). Got {pts.shape}")
    return pts


def make_initial_positions(target_points: np.ndarray, mode: str, offset: float, *, min_spacing: float = 0.0) -> np.ndarray:
    n = len(target_points)
    x_min, y_min = target_points.min(axis=0)
    x_max, y_max = target_points.max(axis=0)
    cx = 0.5 * (x_min + x_max)

    if mode == "hline_below":
        line_y = y_max + offset
        width = (x_max - x_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        width = max(width, req)
        line_x = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n)
        return np.column_stack([line_x, np.full(n, line_y)])

    if mode == "hline_match_targets_x":
        line_y = y_max + offset
        return np.column_stack([target_points[:, 0], np.full(n, line_y)])

    raise ValueError(f"Unknown initial mode: {mode}")


def init_trajectory_linear(p0: np.ndarray, targets: np.ndarray, k_steps: int) -> np.ndarray:
    """Linear interpolation from p0 to targets over k_steps."""
    n = len(p0)
    alphas = np.linspace(0.0, 1.0, int(k_steps))[:, None, None]  # (K,1,1)
    p = (1.0 - alphas) * p0[None, :, :] + alphas * targets[None, :, :]
    return p.astype(np.float64)  # (K,N,2)


def compute_cost_and_grad(
    p: np.ndarray,  # (K,N,2)
    p0: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,2)
    r_safe: float,
    weights: Weights,
    *,
    collision_stride: int = 1,
    eps: float = 1e-9,
) -> tuple[float, np.ndarray]:
    k_steps, n, _ = p.shape
    grad = np.zeros_like(p)
    cost = 0.0

 
    d = p[1:] - p[:-1]  # (K-1,N,2)
    cost += float(weights.w_vel) * float(np.sum(d * d))
    g = 2.0 * float(weights.w_vel) * d
    grad[:-1] -= g
    grad[1:] += g


    if k_steps >= 3 and weights.w_acc > 0:
        a = p[2:] - 2.0 * p[1:-1] + p[:-2]  # (K-2,N,2)
        cost += float(weights.w_acc) * float(np.sum(a * a))
        ga = 2.0 * float(weights.w_acc) * a
        grad[2:] += ga
        grad[1:-1] -= 2.0 * ga
        grad[:-2] += ga


    diff = p[-1] - targets
    cost += float(weights.w_goal) * float(np.sum(diff * diff))
    grad[-1] += 2.0 * float(weights.w_goal) * diff


    if weights.w_col > 0 and r_safe > 0:
        r = float(r_safe)
        w = float(weights.w_col)
        stride = max(1, int(collision_stride))

        for k in range(0, k_steps, stride):
            pk = p[k]  # (N,2)
            dxy = pk[:, None, :] - pk[None, :, :]
            dist2 = np.sum(dxy * dxy, axis=2) + eps
            dist = np.sqrt(dist2)

            mask = dist < r
            np.fill_diagonal(mask, False)
            if not np.any(mask):
                continue

            delta = (r - dist)
            cost += w * float(np.sum((delta[mask]) ** 2))

            factor = np.zeros_like(dist)
            factor[mask] = (-2.0 * w) * (delta[mask] / dist[mask])
            contrib = factor[:, :, None] * dxy  # (N,N,2)
            grad[k] += np.sum(contrib, axis=1)

    grad[0] = 0.0
    return cost, grad


def optimize_trajectories(
    p_init: np.ndarray,
    p0: np.ndarray,
    targets: np.ndarray,
    *,
    r_safe: float,
    weights: Weights,
    iters: int,
    lr: float,
    collision_stride: int,
    print_every: int = 10,
) -> np.ndarray:
    p = p_init.copy()
    p[0] = p0

    for it in range(int(iters)):
        cost, grad = compute_cost_and_grad(
            p,
            p0,
            targets,
            r_safe=float(r_safe),
            weights=weights,
            collision_stride=int(collision_stride),
        )
        p[1:] -= float(lr) * grad[1:]
        p[0] = p0

        if print_every and (it % int(print_every) == 0 or it == int(iters) - 1):
            print(f"iter {it:4d}  cost={cost:,.3f}")

    return p


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_targets = os.path.join(base_dir, "outputs", "target_points.csv")
    default_bg = os.path.join(base_dir, "inputs", "name.png")
    default_out_dir = os.path.join(base_dir, "outputs")

    ap = argparse.ArgumentParser(description="Option 5: trajectory optimization with collision penalties.")
    ap.add_argument("--targets", default=default_targets, help="target_points.csv or target_points.npy")
    ap.add_argument("--bg-image", default=default_bg, help="Optional background image (e.g. name.png)")
    ap.add_argument("--out-dir", default=default_out_dir, help="Output directory for generated files")

    ap.add_argument("--initial", choices=["hline_below", "hline_match_targets_x"], default="hline_below")
    ap.add_argument("--offset", type=float, default=50.0)
    ap.add_argument("--init-min-spacing", type=float, default=25.0)

    ap.add_argument("--k-steps", type=int, default=80, help="Number of trajectory frames (optimization grid)")
    ap.add_argument("--t-end", type=float, default=20.0, help="Only used for animation time scaling")

    ap.add_argument("--r-safe", type=float, default=10.0, help="Minimum separation distance (soft constraint)")
    ap.add_argument("--w-vel", type=float, default=1.0)
    ap.add_argument("--w-acc", type=float, default=0.2)
    ap.add_argument("--w-goal", type=float, default=50.0)
    ap.add_argument("--w-col", type=float, default=50.0)

    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--collision-stride", type=int, default=1, help="Compute collision penalty every N frames (speed)")

    ap.add_argument("--save-npy", action="store_true", help="Save optimal_trajectories.npy")
    ap.add_argument("--save-gif", action="store_true", help="Save optimal_motion.gif")
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-interval-ms", type=int, default=80)
    ap.add_argument("--show", action="store_true", help="Show plots/animation window")
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    _ensure_mpl_configdir_writable()
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    targets = load_target_points(args.targets)
    n = len(targets)

    p0 = make_initial_positions(targets, args.initial, args.offset, min_spacing=float(args.init_min_spacing))
    p_init = init_trajectory_linear(p0, targets, int(args.k_steps))

    weights = Weights(w_vel=args.w_vel, w_acc=args.w_acc, w_goal=args.w_goal, w_col=args.w_col)
    p_opt = optimize_trajectories(
        p_init,
        p0,
        targets,
        r_safe=float(args.r_safe),
        weights=weights,
        iters=int(args.iters),
        lr=float(args.lr),
        collision_stride=int(args.collision_stride),
    )

    if args.save_npy:
        payload = {
            "P": p_opt,
            "targets": targets,
            "initial_positions": p0,
            "params": {
                "r_safe": float(args.r_safe),
                "weights": weights.__dict__,
                "k_steps": int(args.k_steps),
                "t_end": float(args.t_end),
                "iters": int(args.iters),
                "lr": float(args.lr),
                "collision_stride": int(args.collision_stride),
            },
        }
        out_npy = os.path.join(out_dir, "optimal_trajectories.npy")
        np.save(out_npy, payload, allow_pickle=True)
        print(f"Saved: {out_npy}")

    # Plot (static)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    if args.bg_image:
        import cv2

        img = cv2.imread(args.bg_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read bg image: {args.bg_image}")
        h, w = img.shape[:2]
        ax1.imshow(img, cmap="gray", origin="upper", alpha=0.35)
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
    for i in range(n):
        ax1.plot(p_opt[:, i, 0], p_opt[:, i, 1], alpha=0.5, linewidth=0.8)
    ax1.scatter(p0[:, 0], p0[:, 1], c="green", s=30, label="initial", zorder=5)
    ax1.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets", zorder=5)
    ax1.set_title(f"{n} optimized trajectories (direct transcription)")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend()
    fig1.tight_layout()

    # Animation
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if args.bg_image:
        import cv2

        img = cv2.imread(args.bg_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read bg image: {args.bg_image}")
        h, w = img.shape[:2]
        ax2.imshow(img, cmap="gray", origin="upper", alpha=0.35)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)
    ax2.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets")
    drone_dots = ax2.scatter(p0[:, 0], p0[:, 1], c="blue", s=30, label="drones")
    ax2.set_title("Optimized motion (trajectory optimization)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend()
    fig2.tight_layout()

    def update(frame: int):
        k = min(int(frame), p_opt.shape[0] - 1)
        drone_dots.set_offsets(p_opt[k])
        return (drone_dots,)

    ani = FuncAnimation(fig2, update, frames=p_opt.shape[0], interval=int(args.gif_interval_ms), blit=True)
    if args.save_gif:
        out_gif = os.path.join(out_dir, "optimal_motion.gif")
        ani.save(out_gif, writer="pillow", fps=int(args.gif_fps))
        print(f"Saved: {out_gif}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

