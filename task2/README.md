# Task 2 — Transition to New Year Greeting

Goal: move the drone swarm from the **handwritten name** formation (Task 1 final positions) to the greeting **"Happy New Year!"**.

## 0) Generate the greeting image

```bash
python3 task2/generate_greeting_image.py --out task2/inputs/greeting.png
```

## 1) Extract target points for the greeting (reuse Task 1 extractor)

Use the same number of drones as Task 1 (example: `--n 100`):

```bash
python3 task1/extract_target_points.py \
  --image task2/inputs/greeting.png \
  --n 100 --mode skeleton --min-target-spacing 5 \
  --out-dir task2/outputs --debug-png
```

## 2) Generate transition trajectories (BVP shooting)

Start positions are taken from Task 1 final formation (`task1/outputs/target_points.csv`).

```bash
python3 task2/transition.py \
  --start task1/outputs/target_points.csv \
  --targets task2/outputs/target_points.csv \
  --bg-target task2/inputs/greeting.png \
  --model swarm --k-rep 200 --r-safe 12 \
  --k-p 2.0 --k-d 2.5 --v-max 1e9 \
  --t-end 20 --steps 200 \
  --save-gif --save-traj-csv --save-traj-npy --save-traj-plot
```

Outputs are written into `task2/outputs/`.

