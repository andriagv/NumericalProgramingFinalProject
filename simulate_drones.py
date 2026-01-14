import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    m: float = 1.0
    k_p: float = 2.0
    k_d: float = 0.5
    k_rep: float = 5.0
    r_safe: float = 15.0
    v_max: float = 10.0
    # Extra stabilization:
    k_rep_damp: float = 0.0  # relative-velocity damping when drones are too close
    k_vlim: float = 0.0  # smooth speed-limit drag (0 disables)


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


def make_initial_positions(
    target_points: np.ndarray,
    mode: str,
    offset: float,
    *,
    min_spacing: float = 0.0,
) -> np.ndarray:
    n = len(target_points)
    x_min, y_min = target_points.min(axis=0)
    x_max, y_max = target_points.max(axis=0)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    if mode == "hline_above":
        line_y = y_min - offset
        width = (x_max - x_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        width = max(width, req)
        line_x = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n)
        return np.column_stack([line_x, np.full(n, line_y)])
    if mode == "hline_below":
        line_y = y_max + offset
        width = (x_max - x_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        width = max(width, req)
        line_x = np.linspace(cx - 0.5 * width, cx + 0.5 * width, n)
        return np.column_stack([line_x, np.full(n, line_y)])
    if mode == "vline_left":
        line_x = x_min - offset
        height = (y_max - y_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        height = max(height, req)
        line_y = np.linspace(cy - 0.5 * height, cy + 0.5 * height, n)
        return np.column_stack([np.full(n, line_x), line_y])
    if mode == "vline_right":
        line_x = x_max + offset
        height = (y_max - y_min)
        req = max(0.0, float(min_spacing)) * max(0, n - 1)
        height = max(height, req)
        line_y = np.linspace(cy - 0.5 * height, cy + 0.5 * height, n)
        return np.column_stack([np.full(n, line_x), line_y])

    raise ValueError(f"Unknown initial mode: {mode}")


def build_initial_state(initial_positions: np.ndarray) -> np.ndarray:
    n = len(initial_positions)
    state0 = np.zeros(4 * n, dtype=np.float64)
    for i in range(n):
        idx = 4 * i
        state0[idx] = initial_positions[i, 0]
        state0[idx + 1] = initial_positions[i, 1]
    return state0


def drone_ode(t: float, state: np.ndarray, params: Params, targets: np.ndarray) -> np.ndarray:
    n = len(targets)
    dxdt = np.zeros_like(state)

    for i in range(n):
        idx = 4 * i
        x, y, vx, vy = state[idx : idx + 4]
        tx, ty = targets[i]

        # attraction + damping
        f_attr_x = params.k_p * (tx - x)
        f_attr_y = params.k_p * (ty - y)
        f_damp_x = -params.k_d * vx
        f_damp_y = -params.k_d * vy

        # repulsion
        f_rep_x = 0.0
        f_rep_y = 0.0
        f_rep_damp_x = 0.0
        f_rep_damp_y = 0.0
        for j in range(n):
            if i == j:
                continue
            jdx = 4 * j
            xj = state[jdx]
            yj = state[jdx + 1]
            vxj = state[jdx + 2]
            vyj = state[jdx + 3]
            dx = x - xj
            dy = y - yj
            dist = float(np.sqrt(dx * dx + dy * dy + 1e-8))
            if dist < params.r_safe:
                # Barrier-like repulsion (smooth, stronger near 0):
                # U(d) = 0.5*k*(1/d - 1/R)^2 for d < R, else 0
                # F = -∇U = k*(1/d - 1/R)*(1/d^2) * (p_i - p_j)/d
                inv = 1.0 / dist
                coeff = params.k_rep * (inv - (1.0 / params.r_safe)) * (inv**2)
                f_rep_x += coeff * dx * inv
                f_rep_y += coeff * dy * inv

                if params.k_rep_damp > 0:
                    # Dampen relative velocities when close (helps avoid oscillation/crossing)
                    w = (1.0 - dist / params.r_safe)  # 1 at contact, 0 at boundary
                    f_rep_damp_x += -params.k_rep_damp * w * (vx - vxj)
                    f_rep_damp_y += -params.k_rep_damp * w * (vy - vyj)

        # Smooth speed-limit drag (avoids the inconsistent "hard clamp" on vx/vy)
        f_vlim_x = 0.0
        f_vlim_y = 0.0
        if params.k_vlim > 0 and params.v_max > 0:
            speed = float(np.sqrt(vx * vx + vy * vy) + 1e-12)
            excess = max(0.0, speed - params.v_max)
            if excess > 0:
                # Force opposite to velocity, proportional to excess speed
                f_vlim_x = -params.k_vlim * excess * (vx / speed)
                f_vlim_y = -params.k_vlim * excess * (vy / speed)

        f_total_x = f_attr_x + f_damp_x + f_rep_x + f_rep_damp_x + f_vlim_x
        f_total_y = f_attr_y + f_damp_y + f_rep_y + f_rep_damp_y + f_vlim_y

        ax = f_total_x / params.m
        ay = f_total_y / params.m

        dxdt[idx] = vx
        dxdt[idx + 1] = vy
        dxdt[idx + 2] = ax
        dxdt[idx + 3] = ay

    return dxdt


def make_launch_delays(n: int, every: int, delay: float, *, offset: int = 0) -> np.ndarray:
    """
    Create per-drone launch delays.
    Example: every=2, delay=3.0 -> drones 0,2,4,... launch at t=0; drones 1,3,5,... launch at t=3.
    """
    if every <= 0 or delay <= 0:
        return np.zeros(n, dtype=np.float64)
    idx = (np.arange(n, dtype=int) + int(offset)) % int(every)
    return np.where(idx == 0, 0.0, float(delay)).astype(np.float64)


def drone_ode_with_launch_delays(
    t: float,
    state: np.ndarray,
    params: Params,
    targets: np.ndarray,
    launch_delays: np.ndarray,
    *,
    ignore_unlaunched_in_repulsion: bool,
    control: str,
) -> np.ndarray:
    """
    If t < launch_delays[i], drone i is 'not launched':
    - It stays fixed at its initial state (dx=dy=dvx=dvy=0).
    - Optionally, other drones ignore it in repulsion until it launches.
    """
    n = len(targets)
    dxdt = np.zeros_like(state)

    launched = t >= launch_delays

    for i in range(n):
        if not launched[i]:
            continue  # frozen

        idx = 4 * i
        x, y, vx, vy = state[idx : idx + 4]
        tx, ty = targets[i]

        # Repulsion term (same in both control modes)
        f_rep_x = 0.0
        f_rep_y = 0.0
        f_rep_damp_x = 0.0
        f_rep_damp_y = 0.0

        for j in range(n):
            if i == j:
                continue
            if ignore_unlaunched_in_repulsion and (not launched[j]):
                continue
            jdx = 4 * j
            xj = state[jdx]
            yj = state[jdx + 1]
            vxj = state[jdx + 2]
            vyj = state[jdx + 3]
            dx = x - xj
            dy = y - yj
            dist = float(np.sqrt(dx * dx + dy * dy + 1e-8))
            if dist < params.r_safe:
                inv = 1.0 / dist
                coeff = params.k_rep * (inv - (1.0 / params.r_safe)) * (inv**2)
                f_rep_x += coeff * dx * inv
                f_rep_y += coeff * dy * inv

                if params.k_rep_damp > 0:
                    w = (1.0 - dist / params.r_safe)
                    f_rep_damp_x += -params.k_rep_damp * w * (vx - vxj)
                    f_rep_damp_y += -params.k_rep_damp * w * (vy - vyj)

        if control == "first_order":
            # No inertia: command velocity directly toward target + repulsion.
            v_cmd_x = params.k_p * (tx - x) + f_rep_x + f_rep_damp_x
            v_cmd_y = params.k_p * (ty - y) + f_rep_y + f_rep_damp_y

            speed = float(np.sqrt(v_cmd_x * v_cmd_x + v_cmd_y * v_cmd_y) + 1e-12)
            if params.v_max > 0 and speed > params.v_max:
                s = params.v_max / speed
                v_cmd_x *= s
                v_cmd_y *= s

            dxdt[idx] = v_cmd_x
            dxdt[idx + 1] = v_cmd_y
            dxdt[idx + 2] = 0.0
            dxdt[idx + 3] = 0.0
        else:
            # Second-order (mass-spring-damper) dynamics: can overshoot if underdamped.
            f_attr_x = params.k_p * (tx - x)
            f_attr_y = params.k_p * (ty - y)
            f_damp_x = -params.k_d * vx
            f_damp_y = -params.k_d * vy

            f_vlim_x = 0.0
            f_vlim_y = 0.0
            if params.k_vlim > 0 and params.v_max > 0:
                speed = float(np.sqrt(vx * vx + vy * vy) + 1e-12)
                excess = max(0.0, speed - params.v_max)
                if excess > 0:
                    f_vlim_x = -params.k_vlim * excess * (vx / speed)
                    f_vlim_y = -params.k_vlim * excess * (vy / speed)

            f_total_x = f_attr_x + f_damp_x + f_rep_x + f_rep_damp_x + f_vlim_x
            f_total_y = f_attr_y + f_damp_y + f_rep_y + f_rep_damp_y + f_vlim_y

            ax = f_total_x / params.m
            ay = f_total_y / params.m

            dxdt[idx] = vx
            dxdt[idx + 1] = vy
            dxdt[idx + 2] = ax
            dxdt[idx + 3] = ay

    return dxdt


def trajectories_from_solution(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # y: (4N, T) -> positions arrays (N, T)
    n = y.shape[0] // 4
    x = np.zeros((n, y.shape[1]), dtype=np.float64)
    ypos = np.zeros((n, y.shape[1]), dtype=np.float64)
    for i in range(n):
        idx = 4 * i
        x[i] = y[idx]
        ypos[i] = y[idx + 1]
    return x, ypos


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
    Uses O(T*N^2) but N<=~200 is fine for this project.
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
        # ignore diagonal by setting it to +inf
        np.fill_diagonal(d, np.inf)
        # min over all pairs (i,j)
        ij = np.unravel_index(np.argmin(d), d.shape)
        dmin = float(d[ij])
        if dmin < best_d:
            best_d = dmin
            best_i, best_j = int(ij[0]), int(ij[1])
            best_t = float(t[k])

    # normalize so i<j for reporting
    if best_i > best_j:
        best_i, best_j = best_j, best_i
    return best_d, best_i, best_j, best_t


def min_pairwise_distance(points: np.ndarray) -> tuple[float, int, int]:
    """Brute-force minimum pairwise distance among 2D points. Returns (d_min, i, j)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    best = float("inf")
    bi = -1
    bj = -1
    for i in range(n):
        d = pts[i + 1 :] - pts[i]
        if len(d) == 0:
            continue
        dist = np.sqrt((d * d).sum(axis=1))
        jrel = int(np.argmin(dist))
        dmin = float(dist[jrel])
        if dmin < best:
            best = dmin
            bi = i
            bj = i + 1 + jrel
    return best, bi, bj


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate drones moving to target points via IVP and animate.")
    parser.add_argument("--targets", default="target_points.csv", help="target_points.csv or target_points.npy")
    parser.add_argument(
        "--bg-image",
        default=None,
        help="Optional background image (e.g. name.png) to overlay behind points (for validation).",
    )
    parser.add_argument(
        "--initial",
        default="hline_below",
        choices=["hline_above", "hline_below", "vline_left", "vline_right"],
        help="Initial formation shape",
    )
    parser.add_argument("--offset", type=float, default=50.0, help="Offset (pixels) from target bounding box")
    parser.add_argument(
        "--init-min-spacing",
        type=float,
        default=None,
        help="Minimum spacing between neighbors in the initial line (defaults to r_safe).",
    )

    parser.add_argument("--t-end", type=float, default=20.0, help="Simulation end time")
    parser.add_argument("--steps", type=int, default=200, help="Number of evaluation time points")

    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--k-p", type=float, default=2.0)
    parser.add_argument("--k-d", type=float, default=0.5)
    parser.add_argument("--k-rep", type=float, default=5.0)
    parser.add_argument("--r-safe", type=float, default=15.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--k-rep-damp", type=float, default=0.0, help="Extra close-range relative-velocity damping")
    parser.add_argument("--k-vlim", type=float, default=0.0, help="Smooth speed-limit drag strength (0 disables)")

    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)

    parser.add_argument(
        "--control",
        choices=["second_order", "first_order"],
        default="second_order",
        help="Dynamics type: second_order (mass-spring-damper) or first_order (direct velocity to target).",
    )
    parser.add_argument(
        "--stagger-every",
        type=int,
        default=0,
        help="Launch staggering: if >0, every Nth drone launches immediately; others are delayed.",
    )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=0.0,
        help="Launch staggering: delay (seconds) for non-immediate drones.",
    )
    parser.add_argument(
        "--stagger-offset",
        type=int,
        default=0,
        help="Launch staggering: offset applied before modulo (changes which drones are delayed).",
    )
    parser.add_argument(
        "--ignore-unlaunched-in-repulsion",
        action="store_true",
        help="If set, launched drones ignore unlaunched drones in repulsion until they launch.",
    )

    parser.add_argument("--save-traj-csv", action="store_true", help="Save drone_trajectories.csv")
    parser.add_argument("--save-traj-npy", action="store_true", help="Save drone_trajectories.npy (pickle)")
    parser.add_argument("--save-gif", action="store_true", help="Save drone_motion.gif (requires pillow)")
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-interval-ms", type=int, default=50)
    parser.add_argument(
        "--hold-last",
        type=int,
        default=0,
        help="Repeat the last frame this many times (makes the GIF pause at the end).",
    )
    parser.add_argument(
        "--collision-report",
        action="store_true",
        help="Print closest approach distance between any two drones (detects real collisions vs visual crossings).",
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=None,
        help="Collision threshold distance (defaults to r_safe).",
    )
    parser.add_argument("--show", action="store_true", help="Show plots/animation window")
    args = parser.parse_args()

    _ensure_mpl_configdir_writable()
    # In headless environments (like many sandboxes), default GUI backends can crash.
    # Use Agg when we only need to save files.
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.integrate import solve_ivp

    targets = load_target_points(args.targets)
    n = len(targets)

    p = Params(
        m=args.m,
        k_p=args.k_p,
        k_d=args.k_d,
        k_rep=args.k_rep,
        r_safe=args.r_safe,
        v_max=args.v_max,
        k_rep_damp=args.k_rep_damp,
        k_vlim=args.k_vlim,
    )

    # Feasibility warning: if r_safe is larger than target spacing, collisions are unavoidable
    dmin, i_min, j_min = min_pairwise_distance(targets)
    if p.r_safe > dmin:
        print(
            f"WARNING: r_safe={p.r_safe:.3f} > min target spacing={dmin:.3f} "
            f"(targets {i_min} and {j_min}). With fixed targets, some 'collisions' are unavoidable."
        )

    init_spacing = float(p.r_safe) if args.init_min_spacing is None else float(args.init_min_spacing)
    initial_positions = make_initial_positions(targets, args.initial, args.offset, min_spacing=init_spacing)
    state0 = build_initial_state(initial_positions)

    launch_delays = make_launch_delays(n, args.stagger_every, args.stagger_delay, offset=args.stagger_offset)
    if np.any(launch_delays > 0):
        print(
            f"Launch staggering enabled: max_delay={float(launch_delays.max()):.3f}s "
            f"(every={args.stagger_every}, delay={args.stagger_delay})"
        )

    t_eval = np.linspace(0.0, float(args.t_end), int(args.steps))
    sol = solve_ivp(
        fun=lambda t, y: drone_ode_with_launch_delays(
            t,
            y,
            p,
            targets,
            launch_delays,
            ignore_unlaunched_in_repulsion=bool(args.ignore_unlaunched_in_repulsion),
            control=str(args.control),
        ),
        t_span=(0.0, float(args.t_end)),
        y0=state0,
        t_eval=t_eval,
        method="RK45",
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    x_traj, y_traj = trajectories_from_solution(sol.y)

    # Diagnostics: final distance to targets
    final_pos = np.column_stack([x_traj[:, -1], y_traj[:, -1]])
    dist = np.linalg.norm(final_pos - targets, axis=1)
    print(f"N drones: {n}")
    print(f"Final distance to target: mean={dist.mean():.3f}, max={dist.max():.3f}, min={dist.min():.3f}")

    if args.collision_report:
        thr = float(args.collision_threshold) if args.collision_threshold is not None else float(p.r_safe)
        dmin, i, j, tmin = closest_approach(sol.t, x_traj, y_traj)
        status = "COLLISION" if dmin < thr else "no collision"
        print(f"Closest approach: d_min={dmin:.3f} at t={tmin:.3f} between drones ({i},{j}) -> {status} (threshold={thr:.3f})")

    # Trajectory plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    if args.bg_image:
        import cv2

        img = cv2.imread(args.bg_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read bg image: {args.bg_image}")
        h, w = img.shape[:2]
        ax1.imshow(img, cmap="gray", origin="upper", extent=[0, w, h, 0], alpha=0.35)
    for i in range(n):
        ax1.plot(x_traj[i], y_traj[i], alpha=0.5, linewidth=0.8)
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c="green", s=30, label="initial", zorder=5)
    ax1.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets", zorder=5)
    ax1.set_title(f"{n} drone trajectories")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")
    # Pixel coordinates: y grows downward in images. Make plots match image coordinates.
    ax1.invert_yaxis()
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
        ax2.imshow(img, cmap="gray", origin="upper", extent=[0, w, h, 0], alpha=0.35)
    ax2.scatter(targets[:, 0], targets[:, 1], c="red", s=50, label="targets")
    pad = max(50.0, float(args.offset))
    ax2.set_xlim(targets[:, 0].min() - pad, targets[:, 0].max() + pad)
    ax2.set_ylim(targets[:, 1].min() - pad, targets[:, 1].max() + pad)
    ax2.set_title("Drone motion animation")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    ax2.invert_yaxis()
    drone_dots = ax2.scatter(initial_positions[:, 0], initial_positions[:, 1], c="blue", s=30, label="drones")
    ax2.legend()
    fig2.tight_layout()

    def update(frame: int):
        k = min(int(frame), x_traj.shape[1] - 1)
        drone_dots.set_offsets(np.column_stack([x_traj[:, k], y_traj[:, k]]))
        return (drone_dots,)

    total_frames = len(t_eval) + max(0, int(args.hold_last))
    ani = FuncAnimation(fig2, update, frames=total_frames, interval=int(args.gif_interval_ms), blit=True)

    if args.save_gif:
        out_gif = "drone_motion.gif"
        ani.save(out_gif, writer="pillow", fps=int(args.gif_fps))
        print(f"Saved: {out_gif}")

    if args.save_traj_csv:
        out_csv = "drone_trajectories.csv"
        save_trajectories_csv(out_csv, sol.t, x_traj, y_traj)
        print(f"Saved: {out_csv}")

    if args.save_traj_npy:
        out_npy = "drone_trajectories.npy"
        payload = {
            "time": sol.t,
            "x": x_traj,
            "y": y_traj,
            "initial_positions": initial_positions,
            "target_points": targets,
            "params": p.__dict__,
        }
        np.save(out_npy, payload, allow_pickle=True)
        print(f"Saved: {out_npy}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

