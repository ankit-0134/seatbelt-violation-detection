"""
cli.py — Command-line runner (no Streamlit needed)

Usage examples
--------------
# Process a single video
python cli.py --input traffic.mp4

# Custom model paths + line positions (40 % and 70 % of frame height)
python cli.py --input traffic.mp4 \
              --seatbelt models/seatbelt.pt \
              --plate    models/best_plate.pt \
              --line1    0.40 \
              --line2    0.70

# Don't save, just show live preview
python cli.py --input traffic.mp4 --no-save --preview
"""

import argparse
import cv2
import os
import time

from detector import VehicleAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Safety Monitor — CLI")
    p.add_argument("--input",    required=True,           help="Path to input video")
    p.add_argument("--output",   default="output/result.mp4", help="Output video path")
    p.add_argument("--car",      default="models/car_detector.pt", help="Car detection model")
    p.add_argument("--seatbelt", default="models/seatbelt.pt")
    p.add_argument("--plate",    default="models/best_plate.pt")
    p.add_argument("--line1",    type=float, default=0.35, help="Top line (0–1)")
    p.add_argument("--line2",    type=float, default=0.65, help="Bottom line (0–1)")
    p.add_argument("--car-conf", type=float, default=0.40)
    p.add_argument("--sb-conf",  type=float, default=0.35)
    p.add_argument("--pl-conf",  type=float, default=0.35)
    p.add_argument("--no-save",  action="store_true",     help="Skip saving output video")
    p.add_argument("--preview",  action="store_true",     help="Show live OpenCV window")
    return p.parse_args()


def main():
    args = parse_args()

    analyzer = VehicleAnalyzer(
        car_model_path=args.car,
        seatbelt_model_path=args.seatbelt,
        plate_model_path=args.plate,
        line1_y=args.line1,
        line2_y=args.line2,
        car_conf=args.car_conf,
        seatbelt_conf=args.sb_conf,
        plate_conf=args.pl_conf,
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if not args.no_save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"[INFO] Saving to {args.output}")

    totals = {"cars_in_zone": 0, "seatbelt": 0, "no_seatbelt": 0, "plates": 0}
    t0 = time.time()

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        processed, stats = analyzer.process_frame(frame)

        for k in totals:
            totals[k] += stats[k]

        if writer:
            writer.write(processed)

        if args.preview:
            cv2.imshow("Vehicle Safety Monitor  [q to quit]", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

        if (i + 1) % 30 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            print(
                f"  Frame {i+1:>5}/{total}  "
                f"{elapsed:>6.1f}s  "
                f"cars={totals['cars_in_zone']}  "
                f"belt={totals['seatbelt']}  "
                f"no_belt={totals['no_seatbelt']}  "
                f"plates={totals['plates']}"
            )

    cap.release()
    if writer:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print("\n=== FINAL SUMMARY ===")
    for k, v in totals.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'elapsed':<20}: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
