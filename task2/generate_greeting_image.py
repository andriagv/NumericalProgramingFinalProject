import argparse
import os

import cv2
import numpy as np


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(base_dir, "inputs", "greeting.png")

    ap = argparse.ArgumentParser(description='Generate an image with the greeting "Happy New Year!"')
    ap.add_argument("--out", default=default_out, help="Output image path (PNG)")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--bg", type=int, default=255, help="Background grayscale value (0..255)")
    ap.add_argument("--fg", type=int, default=0, help="Text grayscale value (0..255)")

    ap.add_argument("--line1", default="Happy")
    ap.add_argument("--line2", default="New Year!")
    ap.add_argument("--font-scale", type=float, default=6.0)
    ap.add_argument("--thickness", type=int, default=14)
    ap.add_argument("--line-gap", type=int, default=35, help="Vertical gap between lines (pixels)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    img = np.full((int(args.height), int(args.width)), int(args.bg), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put_centered(text: str, y_center: int) -> None:
        (tw, th), baseline = cv2.getTextSize(text, font, float(args.font_scale), int(args.thickness))
        x = int((img.shape[1] - tw) * 0.5)
        # OpenCV uses baseline; this places text approximately centered vertically around y_center.
        y = int(y_center + th * 0.5)
        cv2.putText(
            img,
            text,
            (x, y),
            font,
            float(args.font_scale),
            int(args.fg),
            int(args.thickness),
            lineType=cv2.LINE_AA,
        )

    y_mid = img.shape[0] // 2
    put_centered(str(args.line1), y_mid - int(args.line_gap) - 40)
    put_centered(str(args.line2), y_mid + int(args.line_gap) + 40)

    cv2.imwrite(args.out, img)
    print(f"Saved greeting image: {args.out}")


if __name__ == "__main__":
    main()

