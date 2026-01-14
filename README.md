# Numerical Programming Final Project — Target Points Extraction

This repo includes a small helper script that extracts **equally spaced target points** along the contour of a handwritten/typed name image (e.g. `name.png`).  
It outputs a `target_points` array in **(x, y)** pixel coordinates, ready to be used later as drone targets.

## What it does (high level)

- Loads `name.png`
- Converts to grayscale
- Binarizes with `THRESH_BINARY_INV` so **text becomes white (255)** and background black (0)
- Finds external contours with `cv2.findContours` (**no Canny needed**)
- Samples **N equally spaced points** along the contour(s)
- Saves:
  - `target_points.npy` (NumPy array, shape `(N, 2)`)
  - `target_points.csv` (same points as CSV, header `x,y`)
  - `debug_target_points.png` (optional) for visual checking

## Run

From the project folder:

```bash
python3 extract_target_points.py --image name.png --n 50 --mode interior --debug-png
```

Optional:

```bash
python3 extract_target_points.py --image name.png --n 50 --mode interior --show
```

Point size tweaks:

```bash
python3 extract_target_points.py --image name.png --n 50 --mode interior --min-border-dist 6 --debug-png --debug-point-radius 4
python3 extract_target_points.py --image name.png --n 50 --mode interior --show --show-point-size 90
```

Better readability per letter (recommended):

```bash
python3 extract_target_points.py --image name.png --n 50 --mode interior --min-per-component 8 --debug-png
python3 extract_target_points.py --image name.png --n 50 --mode interior --extreme-dirs 8 --debug-png
```

Medial axis (skeleton) mode (points lie on the centerline):

```bash
python3 extract_target_points.py --image name.png --n 50 --mode skeleton --min-per-component 8 --debug-png
```

## Drone simulation (IVP + animation)

Run a simple IVP model (attraction + damping + collision avoidance) and animate:

```bash
python3 simulate_drones.py --targets target_points.csv --initial hline_below --offset 50 --t-end 20 --steps 200 --show
```

Save a GIF + trajectories:

```bash
python3 simulate_drones.py --targets target_points.csv --initial hline_below --offset 50 --t-end 20 --steps 200 --save-gif --save-traj-csv --save-traj-npy
```

## Outputs

- `target_points.npy`
- `target_points.csv`
- `debug_target_points.png` (only if you pass `--debug-png`)

