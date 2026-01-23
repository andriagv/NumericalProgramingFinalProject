import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """
    Parameters for the single-drone second-order model used by BVP shooting.
    """

    m: float = 1.0
    k_p: float = 2.0
    k_d: float = 0.5


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


def _saturate(v: np.ndarray, v_max: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    vmax = float(v_max)
    if vmax <= 0 or not np.isfinite(vmax):
        return v
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.ones_like(n)
    mask = n > vmax
    scale[mask] = vmax / (n[mask] + 1e-12)
    return v * scale


def _repulsive_force(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    n = len(positions)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    k = float(k_rep)
    r = float(r_safe)
    if k <= 0 or r <= 0:
        return np.zeros_like(positions, dtype=np.float64)
    diff = positions[:, None, :] - positions[None, :, :]  # (N,N,2)
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    mask = dist2 < r * r
    if not np.any(mask):
        return np.zeros_like(positions, dtype=np.float64)
    dist = np.sqrt(dist2 + 1e-12)
    force = k * diff / (dist[:, :, None] ** 3)
    force[~mask] = 0.0
    return np.sum(force, axis=1)


def make_initial_positions(
    target_points: np.ndarray,
    mode: str,
    offset: float,
    *,
    min_spacing: float = 25.0,
    two_line_gap: float = 30.0,
) -> np.ndarray:
    """
    Still initial configuration of the swarm in 2D pixel coordinates.
    """
    n = len(target_points)
    x_min, y_min = target_points.min(axis=0)
    x_max, y_max = target_points.max(axis=0)
    cx = 0.5 * (x_min + x_max)

    if mode == "hline_match_targets_x":
        line_y = y_max + offset
        return np.column_stack([target_points[:, 0], np.full(n, line_y)])

    if mode == "two_hlines_below":
        n1 = (n + 1) // 2
        n2 = n - n1
        base_y = y_max + offset
        y1 = base_y
        y2 = base_y + float(two_line_gap)

        width = float(x_max - x_min)
        req1 = max(0.0, float(min_spacing)) * max(0, n1 - 1)
        req2 = max(0.0, float(min_spacing)) * max(0, n2 - 1)
        width = max(width, req1, req2)

        x1 = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n1) if n1 > 0 else np.zeros((0,))
        x2 = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n2) if n2 > 0 else np.zeros((0,))
        p1 = np.column_stack([x1, np.full(n1, y1)]) if n1 > 0 else np.zeros((0, 2))
        p2 = np.column_stack([x2, np.full(n2, y2)]) if n2 > 0 else np.zeros((0, 2))
        return np.vstack([p1, p2])

    if mode == "hline_below":
        line_y = y_max + offset
        width = float(x_max - x_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        width = max(width, req)
        line_x = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n)
        return np.column_stack([line_x, np.full(n, line_y)])

    raise ValueError(f"Unknown initial mode: {mode}")


def drone_ode_single_second_order(t: float, state: np.ndarray, params: Params, target_xy: np.ndarray) -> np.ndarray:
    """
    Single-drone second-order model:
      x' = v
      v' = (k_p*(T-x) - k_d*v)/m
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    tx, ty = float(target_xy[0]), float(target_xy[1])

    f_attr_x = params.k_p * (tx - x)
    f_attr_y = params.k_p * (ty - y)
    f_damp_x = -params.k_d * vx
    f_damp_y = -params.k_d * vy

    ax = (f_attr_x + f_damp_x) / params.m
    ay = (f_attr_y + f_damp_y) / params.m
    return np.array([vx, vy, ax, ay], dtype=np.float64)


def solve_bvp_shooting_all(
    *,
    t_eval: np.ndarray,
    initial_positions: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,2)
    params: Params,
    rtol: float,
    atol: float,
    match_final_velocity: bool = False,
    final_velocity_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BVP via (single) shooting, solved independently for each drone:
      Find v0=(vx0,vy0) so that position at t_end is close to target.

    If match_final_velocity=True, solve a least-squares problem that also prefers v(T)≈0,
    which reduces end oscillations.

    Returns x_traj,y_traj with shape (N,T).
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares, root

    t_eval = np.asarray(t_eval, dtype=np.float64)
    t0 = float(t_eval[0])
    t1 = float(t_eval[-1])
    if t1 <= t0:
        raise ValueError("t_eval must span a positive interval")

    n = len(targets)
    x_traj = np.zeros((n, len(t_eval)), dtype=np.float64)
    y_traj = np.zeros((n, len(t_eval)), dtype=np.float64)

    for i in range(n):
        p0 = initial_positions[i].astype(np.float64)
        tg = targets[i].astype(np.float64)

        # Initial guess: constant-velocity to target over [t0,t1]
        v_guess = (tg - p0) / max(1e-9, (t1 - t0))

        def F_pos(v0: np.ndarray) -> np.ndarray:
            y0 = np.array([p0[0], p0[1], float(v0[0]), float(v0[1])], dtype=np.float64)
            sol = solve_ivp(
                fun=lambda t, y: drone_ode_single_second_order(t, y, params, tg),
                t_span=(t0, t1),
                y0=y0,
                t_eval=None,
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
            xf, yf = sol.y[0, -1], sol.y[1, -1]
            return np.array([xf - tg[0], yf - tg[1]], dtype=np.float64)

        def F_pos_vel(v0: np.ndarray) -> np.ndarray:
            y0 = np.array([p0[0], p0[1], float(v0[0]), float(v0[1])], dtype=np.float64)
            sol = solve_ivp(
                fun=lambda t, y: drone_ode_single_second_order(t, y, params, tg),
                t_span=(t0, t1),
                y0=y0,
                t_eval=None,
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
            xf, yf, vxf, vyf = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], sol.y[3, -1]
            wv = float(final_velocity_weight)
            return np.array([xf - tg[0], yf - tg[1], wv * vxf, wv * vyf], dtype=np.float64)

        if match_final_velocity:
            res = least_squares(F_pos_vel, x0=v_guess, method="trf")
            v0 = res.x if res.success else v_guess
        else:
            res = root(F_pos, x0=v_guess, method="hybr")
            v0 = res.x if res.success else v_guess

        y0 = np.array([p0[0], p0[1], float(v0[0]), float(v0[1])], dtype=np.float64)
        sol = solve_ivp(
            fun=lambda t, y: drone_ode_single_second_order(t, y, params, tg),
            t_span=(t0, t1),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        x_traj[i] = sol.y[0]
        y_traj[i] = sol.y[1]

    return x_traj, y_traj


def solve_swarm_ivp(
    *,
    t_eval: np.ndarray,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    targets: np.ndarray,
    params: Params,
    rtol: float,
    atol: float,
    k_rep: float,
    r_safe: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.integrate import solve_ivp

    t_eval = np.asarray(t_eval, dtype=np.float64)
    n = len(targets)
    if n == 0:
        return np.zeros((0, len(t_eval))), np.zeros((0, len(t_eval)))

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        pos = state[: 2 * n].reshape(n, 2)
        vel = state[2 * n :].reshape(n, 2)
        rep = _repulsive_force(pos, k_rep=k_rep, r_safe=r_safe)
        acc = (params.k_p * (targets - pos) + rep - params.k_d * vel) / params.m
        xdot = _saturate(vel, v_max)
        return np.concatenate([xdot.reshape(-1), acc.reshape(-1)])

    y0 = np.concatenate([initial_positions.reshape(-1), initial_velocities.reshape(-1)])
    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    pos_hist = sol.y[: 2 * n].reshape(n, 2, len(t_eval))
    x_traj = pos_hist[:, 0, :]
    y_traj = pos_hist[:, 1, :]
    return x_traj, y_traj


def save_trajectories_csv(path: str, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    n, tt = x.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("drone_id,time,x,y\n")
        for i in range(n):
            for k in range(tt):
                f.write(f"{i},{t[k]:.6f},{x[i, k]:.6f},{y[i, k]:.6f}\n")


def closest_approach(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, int, int, float]:
    """
    Return (min_distance, i, j, t_at_min) over all pairs i<j across all time steps.
    Diagnostic only: BVP shooting here is independent per drone (no collision avoidance).
    """
    n, tt = x.shape
    best_d = float("inf")
    best_i = -1
    best_j = -1
    best_t = float(t[0]) if len(t) else 0.0

    for k in range(tt):
        px = x[:, k][:, None]
        py = y[:, k][:, None]
        dx = px - px.T
        dy = py - py.T
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        ij = np.unravel_index(np.argmin(d), d.shape)
        dmin = float(d[ij])
        if dmin < best_d:
            best_d = dmin
            best_i, best_j = int(ij[0]), int(ij[1])
            best_t = float(t[k])

    if best_i > best_j:
        best_i, best_j = best_j, best_i
    return best_d, best_i, best_j, best_t


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_targets = os.path.join(base_dir, "outputs", "target_points.csv")
    default_bg = os.path.join(base_dir, "inputs", "name.png")
    default_out_dir = os.path.join(base_dir, "outputs")

    parser = argparse.ArgumentParser(
        description="Task 1 trajectories + visualization (BVP shooting or swarm IVP with repulsion)."
    )
    parser.add_argument("--targets", default=default_targets, help="target_points.csv or target_points.npy")
    parser.add_argument("--bg-image", default=default_bg, help="Optional background image (e.g. name.png)")
    parser.add_argument("--out-dir", default=default_out_dir, help="Output directory for generated files")

    parser.add_argument(
        "--initial",
        default="hline_below",
        choices=["hline_below", "hline_match_targets_x", "two_hlines_below"],
        help="Still initial formation shape",
    )
    parser.add_argument("--offset", type=float, default=50.0, help="Offset (pixels) from target bounding box")
    parser.add_argument(
        "--two-line-gap",
        type=float,
        default=30.0,
        help="(initial=two_hlines_below) Vertical gap between the two starting lines (pixels).",
    )
    parser.add_argument("--init-min-spacing", type=float, default=25.0, help="Min spacing in initial formation (pixels)")

    parser.add_argument("--t-end", type=float, default=20.0, help="End time T for BVP")
    parser.add_argument("--steps", type=int, default=200, help="Number of evaluation time points")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)

    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--k-p", type=float, default=2.0)
    parser.add_argument("--k-d", type=float, default=0.5)
    parser.add_argument(
        "--model",
        choices=["swarm", "shooting"],
        default="swarm",
        help="swarm=IVP with repulsion; shooting=per-drone BVP (no collision avoidance).",
    )
    parser.add_argument("--k-rep", type=float, default=200.0, help="Repulsion gain for swarm model")
    parser.add_argument("--r-safe", type=float, default=12.0, help="Safety radius for repulsion (pixels)")
    parser.add_argument("--v-max", type=float, default=1e9, help="Velocity saturation (pixels/sec)")

    parser.add_argument(
        "--bvp-match-final-velocity",
        action="store_true",
        help="Also try to make final velocity close to 0 (least-squares), reducing end oscillations.",
    )
    parser.add_argument(
        "--bvp-final-velocity-weight",
        type=float,
        default=1.0,
        help="Weight for final velocity residual when --bvp-match-final-velocity is set.",
    )

    parser.add_argument("--save-traj-csv", action="store_true", help="Save drone_trajectories.csv")
    parser.add_argument("--save-traj-npy", action="store_true", help="Save drone_trajectories.npy (pickle)")
    parser.add_argument("--save-traj-plot", action="store_true", help="Save drone_trajectories.png")

    parser.add_argument("--save-gif", action="store_true", help="Save drone_motion.gif (requires pillow)")
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-interval-ms", type=int, default=50)
    parser.add_argument("--hold-last", type=int, default=0, help="Repeat the last frame this many times (pause at end)")
    parser.add_argument("--drone-size", type=float, default=30.0, help="Drone marker size (scatter)")
    parser.add_argument("--target-size", type=float, default=50.0, help="Target marker size (scatter)")
    parser.add_argument("--initial-size", type=float, default=30.0, help="Initial marker size (scatter)")

    parser.add_argument(
        "--collision-report",
        action="store_true",
        help="Print closest-approach distance (diagnostic; shooting has no collision avoidance).",
    )
    parser.add_argument("--show", action="store_true", help="Show plots/animation window")
    args = parser.parse_args()

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

    params = Params(m=float(args.m), k_p=float(args.k_p), k_d=float(args.k_d))

    initial_positions = make_initial_positions(
        targets,
        args.initial,
        float(args.offset),
        min_spacing=float(args.init_min_spacing),
        two_line_gap=float(args.two_line_gap),
    )

    t_eval = np.linspace(0.0, float(args.t_end), int(args.steps))
    if args.model == "shooting":
        x_traj, y_traj = solve_bvp_shooting_all(
            t_eval=t_eval,
            initial_positions=initial_positions,
            targets=targets,
            params=params,
            rtol=float(args.rtol),
            atol=float(args.atol),
            match_final_velocity=bool(args.bvp_match_final_velocity),
            final_velocity_weight=float(args.bvp_final_velocity_weight),
        )
    else:
        v0 = np.zeros_like(initial_positions)
        x_traj, y_traj = solve_swarm_ivp(
            t_eval=t_eval,
            initial_positions=initial_positions,
            initial_velocities=v0,
            targets=targets,
            params=params,
            rtol=float(args.rtol),
            atol=float(args.atol),
            k_rep=float(args.k_rep),
            r_safe=float(args.r_safe),
            v_max=float(args.v_max),
        )

    final_pos = np.column_stack([x_traj[:, -1], y_traj[:, -1]])
    dist = np.linalg.norm(final_pos - targets, axis=1)
    print(f"N drones: {n}")
    print(f"Final distance to target: mean={dist.mean():.3f}, max={dist.max():.3f}, min={dist.min():.3f}")

    if args.collision_report:
        dmin, i, j, tmin = closest_approach(t_eval, x_traj, y_traj)
        print(f"Closest approach (diagnostic): d_min={dmin:.3f} at t={tmin:.3f} between drones ({i},{j})")

    # Trajectory plot
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
        ax1.plot(x_traj[i], y_traj[i], alpha=0.5, linewidth=0.8)
    ax1.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        c="green",
        s=float(args.initial_size),
        label="initial",
        zorder=5,
    )
    ax1.scatter(
        targets[:, 0],
        targets[:, 1],
        c="red",
        s=float(args.target_size),
        label="targets",
        zorder=5,
    )
    title_suffix = "BVP shooting" if args.model == "shooting" else "swarm IVP + repulsion"
    ax1.set_title(f"{n} drone trajectories ({title_suffix})")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")
    if not args.bg_image:
        ax1.invert_yaxis()
    ax1.legend()
    fig1.tight_layout()

    if args.save_traj_plot:
        out_png = os.path.join(out_dir, "drone_trajectories.png")
        fig1.savefig(out_png, dpi=200)
        print(f"Saved: {out_png}")

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
    ax2.scatter(targets[:, 0], targets[:, 1], c="red", s=float(args.target_size), label="targets")
    pad = max(50.0, float(args.offset))
    if not args.bg_image:
        ax2.set_xlim(targets[:, 0].min() - pad, targets[:, 0].max() + pad)
        ax2.set_ylim(targets[:, 1].max() + pad, targets[:, 1].min() - pad)
    ax2.set_title(f"Drone motion animation ({title_suffix})")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    if not args.bg_image:
        ax2.invert_yaxis()
    drone_dots = ax2.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        c="blue",
        s=float(args.drone_size),
        label="drones",
    )
    ax2.legend()
    fig2.tight_layout()

    def update(frame: int):
        k = min(int(frame), x_traj.shape[1] - 1)
        drone_dots.set_offsets(np.column_stack([x_traj[:, k], y_traj[:, k]]))
        return (drone_dots,)

    total_frames = len(t_eval) + max(0, int(args.hold_last))
    ani = FuncAnimation(fig2, update, frames=total_frames, interval=int(args.gif_interval_ms), blit=True)

    if args.save_gif:
        out_gif = os.path.join(out_dir, "drone_motion.gif")
        ani.save(out_gif, writer="pillow", fps=int(args.gif_fps))
        print(f"Saved: {out_gif}")

    if args.save_traj_csv:
        out_csv = os.path.join(out_dir, "drone_trajectories.csv")
        save_trajectories_csv(out_csv, t_eval, x_traj, y_traj)
        print(f"Saved: {out_csv}")

    if args.save_traj_npy:
        out_npy = os.path.join(out_dir, "drone_trajectories.npy")
        payload = {
            "time": t_eval,
            "x": x_traj,
            "y": y_traj,
            "initial_positions": initial_positions,
            "target_points": targets,
            "params": params.__dict__,
            "bvp": {
                "match_final_velocity": bool(args.bvp_match_final_velocity),
                "final_velocity_weight": float(args.bvp_final_velocity_weight),
            },
        }
        np.save(out_npy, payload, allow_pickle=True)
        print(f"Saved: {out_npy}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

