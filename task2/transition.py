import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    m: float = 1.0
    k_p: float = 2.0
    k_d: float = 0.5


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


def drone_ode_single_second_order(t: float, state: np.ndarray, params: Params, target_xy: np.ndarray) -> np.ndarray:
    x, y, vx, vy = state
    tx, ty = float(target_xy[0]), float(target_xy[1])
    ax = (params.k_p * (tx - x) - params.k_d * vx) / params.m
    ay = (params.k_p * (ty - y) - params.k_d * vy) / params.m
    return np.array([vx, vy, ax, ay], dtype=np.float64)


def solve_bvp_shooting_all(
    *,
    t_eval: np.ndarray,
    start_positions: np.ndarray,  # (N,2)
    targets: np.ndarray,  # (N,2)
    params: Params,
    rtol: float,
    atol: float,
    match_final_velocity: bool,
    final_velocity_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-drone BVP via shooting: choose v(0) so that x(T)≈target (and optionally v(T)≈0).
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares, root

    t_eval = np.asarray(t_eval, dtype=np.float64)
    t0 = float(t_eval[0])
    t1 = float(t_eval[-1])

    n = len(targets)
    x_traj = np.zeros((n, len(t_eval)), dtype=np.float64)
    y_traj = np.zeros((n, len(t_eval)), dtype=np.float64)

    for i in range(n):
        p0 = start_positions[i].astype(np.float64)
        tg = targets[i].astype(np.float64)
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
    start_positions: np.ndarray,
    start_velocities: np.ndarray,
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

    y0 = np.concatenate([start_positions.reshape(-1), start_velocities.reshape(-1)])
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
    Return (min_distance, i, j, t_at_min) over all pairs across all time steps.
    Note: this is only a diagnostic. Task 2 (shooting) has no inter-drone repulsion,
    so it does not guarantee collision-free motion.
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
    default_out_dir = os.path.join(base_dir, "outputs")

    ap = argparse.ArgumentParser(
        description="Task 2: transition from handwritten name to holiday greeting (BVP shooting or swarm IVP)."
    )
    ap.add_argument("--start", required=True, help="Start positions (Task 1 final formation), csv/npy (N,2)")
    ap.add_argument("--targets", required=True, help="Target positions for greeting, csv/npy (N,2)")
    ap.add_argument("--bg-target", default=None, help="Optional greeting image to show behind targets")
    ap.add_argument("--out-dir", default=default_out_dir, help="Output directory")

    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)

    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--k-p", type=float, default=2.0)
    ap.add_argument("--k-d", type=float, default=0.5)
    ap.add_argument(
        "--model",
        choices=["swarm", "shooting"],
        default="swarm",
        help="swarm=IVP with repulsion; shooting=per-drone BVP (no collision avoidance).",
    )
    ap.add_argument("--k-rep", type=float, default=200.0, help="Repulsion gain for swarm model")
    ap.add_argument("--r-safe", type=float, default=12.0, help="Safety radius for repulsion (pixels)")
    ap.add_argument("--v-max", type=float, default=1e9, help="Velocity saturation (pixels/sec)")
    ap.add_argument("--bvp-match-final-velocity", action="store_true")
    ap.add_argument("--bvp-final-velocity-weight", type=float, default=1.0)

    ap.add_argument("--save-traj-csv", action="store_true")
    ap.add_argument("--save-traj-npy", action="store_true")
    ap.add_argument("--save-traj-plot", action="store_true")
    ap.add_argument("--save-gif", action="store_true")
    ap.add_argument("--gif-fps", type=int, default=20)
    ap.add_argument("--gif-interval-ms", type=int, default=50)
    ap.add_argument("--hold-last", type=int, default=0)
    ap.add_argument("--collision-report", action="store_true", help="Report closest approach distance between any two drones")
    ap.add_argument(
        "--collision-threshold",
        type=float,
        default=10.0,
        help="Collision threshold distance in pixels (used only for reporting)",
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    start = load_points(args.start)
    targets = load_points(args.targets)
    if len(start) != len(targets):
        raise ValueError(
            f"N mismatch: start has {len(start)} points, targets has {len(targets)} points. "
            "Regenerate target_points for the greeting with the SAME --n as Task 1."
        )

    t_eval = np.linspace(0.0, float(args.t_end), int(args.steps))
    params = Params(m=float(args.m), k_p=float(args.k_p), k_d=float(args.k_d))

    _ensure_mpl_configdir_writable()
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if args.model == "swarm":
        v0 = np.zeros_like(start)
        x_traj, y_traj = solve_swarm_ivp(
            t_eval=t_eval,
            start_positions=start,
            start_velocities=v0,
            targets=targets,
            params=params,
            rtol=float(args.rtol),
            atol=float(args.atol),
            k_rep=float(args.k_rep),
            r_safe=float(args.r_safe),
            v_max=float(args.v_max),
        )
    else:
        x_traj, y_traj = solve_bvp_shooting_all(
            t_eval=t_eval,
            start_positions=start,
            targets=targets,
            params=params,
            rtol=float(args.rtol),
            atol=float(args.atol),
            match_final_velocity=bool(args.bvp_match_final_velocity),
            final_velocity_weight=float(args.bvp_final_velocity_weight),
        )

    final_pos = np.column_stack([x_traj[:, -1], y_traj[:, -1]])
    dist = np.linalg.norm(final_pos - targets, axis=1)
    print(f"N drones: {len(targets)}")
    print(f"Final distance to target: mean={dist.mean():.3f}, max={dist.max():.3f}, min={dist.min():.3f}")

    if args.collision_report:
        dmin, i, j, tmin = closest_approach(t_eval, x_traj, y_traj)
        thr = float(args.collision_threshold)
        status = "COLLISION" if dmin < thr else "no collision"
        print(f"Closest approach: d_min={dmin:.3f} at t={tmin:.3f} between drones ({i},{j}) -> {status} (threshold={thr:.3f})")

    # Plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    if args.bg_target:
        import cv2

        img = cv2.imread(args.bg_target, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read bg image: {args.bg_target}")
        h, w = img.shape[:2]
        ax1.imshow(img, cmap="gray", origin="upper", alpha=0.35)
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
    for i in range(len(targets)):
        ax1.plot(x_traj[i], y_traj[i], alpha=0.5, linewidth=0.8)
    ax1.scatter(start[:, 0], start[:, 1], c="green", s=30, label="start (name)", zorder=5)
    ax1.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets (greeting)", zorder=5)
    title_suffix = "BVP shooting" if args.model == "shooting" else "swarm IVP + repulsion"
    ax1.set_title(f"Task 2 transition trajectories ({title_suffix})")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")
    if not args.bg_target:
        ax1.invert_yaxis()
    ax1.legend()
    fig1.tight_layout()

    if args.save_traj_plot:
        out_png = os.path.join(out_dir, "transition_trajectories.png")
        fig1.savefig(out_png, dpi=200)
        print(f"Saved: {out_png}")

    # Animation
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if args.bg_target:
        import cv2

        img = cv2.imread(args.bg_target, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read bg image: {args.bg_target}")
        h, w = img.shape[:2]
        ax2.imshow(img, cmap="gray", origin="upper", alpha=0.35)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)
    ax2.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets")
    drone_dots = ax2.scatter(start[:, 0], start[:, 1], c="blue", s=30, label="drones")
    ax2.set_title(f"Task 2 transition animation ({title_suffix})")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    if not args.bg_target:
        ax2.invert_yaxis()
    ax2.legend()
    fig2.tight_layout()

    def update(frame: int):
        k = min(int(frame), x_traj.shape[1] - 1)
        drone_dots.set_offsets(np.column_stack([x_traj[:, k], y_traj[:, k]]))
        return (drone_dots,)

    total_frames = len(t_eval) + max(0, int(args.hold_last))
    ani = FuncAnimation(fig2, update, frames=total_frames, interval=int(args.gif_interval_ms), blit=True)

    if args.save_gif:
        out_gif = os.path.join(out_dir, "transition_motion.gif")
        ani.save(out_gif, writer="pillow", fps=int(args.gif_fps))
        print(f"Saved: {out_gif}")

    if args.save_traj_csv:
        out_csv = os.path.join(out_dir, "transition_trajectories.csv")
        save_trajectories_csv(out_csv, t_eval, x_traj, y_traj)
        print(f"Saved: {out_csv}")

    if args.save_traj_npy:
        out_npy = os.path.join(out_dir, "transition_trajectories.npy")
        payload = {
            "time": t_eval,
            "x": x_traj,
            "y": y_traj,
            "start_positions": start,
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

