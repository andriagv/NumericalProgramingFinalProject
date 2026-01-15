# Report (Task 1) — LaTeX

This folder contains the LaTeX source for **Task 1: Static Formation on a Handwritten Input**.

## Compile

From the project root:

```bash
cd task1/report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Notes:
- The report can reference Task 1 images using paths like `../inputs/name.png` or `../outputs/debug_target_points.png`.
- If some images are missing (because you did not generate them), just comment out the corresponding `\includegraphics{...}` lines in `main.tex`.

