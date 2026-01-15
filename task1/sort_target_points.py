import argparse
import os

import numpy as np


def load_csv(path: str) -> np.ndarray:
    pts = np.loadtxt(path, delimiter=",", skiprows=1)
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points in {path}, got {pts.shape}")
    return pts


def save_csv(path: str, pts: np.ndarray) -> None:
    pts = np.asarray(pts)
    fmt = "%d" if np.issubdtype(pts.dtype, np.integer) else "%.6f"
    np.savetxt(path, pts, fmt=fmt, delimiter=",", header="x,y", comments="")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "outputs", "target_points.csv")
    default_npy = os.path.join(base_dir, "outputs", "target_points.npy")

    p = argparse.ArgumentParser(description="Sort target_points by x asc, then y asc.")
    p.add_argument("--csv", default=default_csv, help="Path to target_points.csv")
    p.add_argument("--npy", default=default_npy, help="Path to target_points.npy (will be overwritten)")
    p.add_argument(
        "--order",
        default="xy",
        choices=["xy", "y_asc", "y_desc"],
        help="Sort order: xy = x asc then y asc; y_asc = y asc then x asc; y_desc = y desc then x asc",
    )
    args = p.parse_args()

    csv_path = args.csv
    pts = load_csv(csv_path)

    if args.order == "y_desc":
        # Primary y descending, secondary x ascending
        order = np.lexsort((pts[:, 0], -pts[:, 1]))
    elif args.order == "y_asc":
        # Primary y ascending, secondary x ascending
        order = np.lexsort((pts[:, 0], pts[:, 1]))
    else:
        # Primary x ascending, secondary y ascending
        order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_sorted = pts[order]

    save_csv(csv_path, pts_sorted)

    # Keep .npy in sync if it exists or if a path is provided
    if args.npy:
        np.save(args.npy, pts_sorted)

    print(f"Sorted {len(pts_sorted)} points and saved: {csv_path}")
    if args.npy:
        print(f"Updated: {args.npy}")
    print("First 5 points:")
    print(pts_sorted[:5])


if __name__ == "__main__":
    main()

