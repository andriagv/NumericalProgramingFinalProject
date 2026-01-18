# Report (Task 2) — LaTeX

This folder contains the LaTeX source for **Task 2: Transition to New Year Greeting**.

## Compile

From the project root:

```bash
cd task2/report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Notes:
- The report can reference Task 2 images using paths like `../inputs/greeting.png` or `../outputs/transition_motion.gif`.
- If some images are missing (because you did not generate them), comment out the corresponding `\includegraphics{...}` lines in `main.tex`.

