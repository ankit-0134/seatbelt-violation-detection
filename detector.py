import cv2
import numpy as np
from ultralytics import YOLO
import os


class VehicleAnalyzer:
    def __init__(
        self,
        car_model_path: str = "models/car_detector.pt",
        seatbelt_model_path: str = "models/seatbelt.pt",
        plate_model_path: str = "models/best_plate.pt",
        line1_y: float = 0.35,
        line2_y: float = 0.65,
        car_conf: float = 0.4,
        seatbelt_conf: float = 0.35,
        plate_conf: float = 0.35,
    ):
        """
        Args:
            car_model_path   : Custom YOLOv8 car detector (car_detector.pt)
            seatbelt_model_path: seatbelt.pt — detects seatbelt / no-seatbelt
            plate_model_path : best_plate.pt — detects number plates
            line1_y: Top detection line (0.0–1.0 of frame height)
            line2_y: Bottom detection line (0.0–1.0 of frame height)
        """
        self.line1_y = line1_y
        self.line2_y = line2_y
        self.car_conf = car_conf
        self.seatbelt_conf = seatbelt_conf
        self.plate_conf = plate_conf

        print("[INFO] Loading models …")
        self.car_model      = YOLO(car_model_path)
        self.seatbelt_model = YOLO(seatbelt_model_path)
        self.plate_model    = YOLO(plate_model_path)
        print(f"[INFO] car_detector classes : {self.car_model.names}")
        print("[INFO] All models loaded.")

        # car_detector.pt is a custom model — accept ALL its classes as vehicles.
        # We read the class names directly from the model so nothing is hard-coded.
        self.car_class_names = self.car_model.names   # {0: 'car', 1: 'truck', …}

        # Colours
        self.CLR_LINE = (0, 255, 255)       # cyan lines
        self.CLR_CAR = (255, 165, 0)         # orange car box
        self.CLR_SEATBELT = (0, 255, 0)      # green seatbelt
        self.CLR_NO_SEATBELT = (0, 0, 255)   # red no-seatbelt
        self.CLR_PLATE = (255, 255, 0)        # yellow plate
        self.CLR_TEXT_BG = (20, 20, 20)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _box_center_y(self, box) -> float:
        return float((box[1] + box[3]) / 2)

    def _car_in_zone(self, box, h: int) -> bool:
        """True when the car's centre falls between the two lines."""
        cy = self._box_center_y(box)
        y1 = self.line1_y * h
        y2 = self.line2_y * h
        return y1 <= cy <= y2

    def _draw_lines(self, frame):
        h, w = frame.shape[:2]
        y1 = int(self.line1_y * h)
        y2 = int(self.line2_y * h)

        # dashed line helper
        def dashed_line(img, y, color, thickness=2, dash=20, gap=15):
            for x in range(0, w, dash + gap):
                cv2.line(img, (x, y), (min(x + dash, w), y), color, thickness)

        dashed_line(frame, y1, self.CLR_LINE, thickness=2)
        dashed_line(frame, y2, self.CLR_LINE, thickness=2)

        # Labels on the right edge
        cv2.putText(frame, f"LINE 1  ({self.line1_y:.2f})", (w - 200, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_LINE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"LINE 2  ({self.line2_y:.2f})", (w - 200, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.CLR_LINE, 1, cv2.LINE_AA)

        # Shaded detection zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

    def _label(self, frame, text: str, x: int, y: int, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), self.CLR_TEXT_BG, -1)
        cv2.putText(frame, text, (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        stats = {"cars_in_zone": 0, "seatbelt": 0, "no_seatbelt": 0, "plates": 0}

        # 1) Draw detection zone lines
        self._draw_lines(frame)

        # 2) Detect vehicles in full frame
        car_results = self.car_model(frame, conf=self.car_conf, verbose=False)[0]

        for det in car_results.boxes:
            cls_id   = int(det.cls[0])
            cls_name = self.car_class_names.get(cls_id, f"cls{cls_id}")
            bx1, by1, bx2, by2 = map(int, det.xyxy[0])
            conf_car = float(det.conf[0])

            # Draw car box always
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.CLR_CAR, 2)
            self._label(frame, f"{cls_name} {conf_car:.2f}", bx1, by1, self.CLR_CAR)

            # Only analyse cars whose centre is inside the zone
            if not self._car_in_zone(det.xyxy[0], h):
                continue

            stats["cars_in_zone"] += 1

            # Crop the vehicle ROI (with small padding)
            pad = 10
            rx1 = max(0, bx1 - pad)
            ry1 = max(0, by1 - pad)
            rx2 = min(w, bx2 + pad)
            ry2 = min(h, by2 + pad)
            roi = frame[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            # 3) Seatbelt detection on ROI
            sb_results = self.seatbelt_model(roi, conf=self.seatbelt_conf, verbose=False)[0]
            for sb in sb_results.boxes:
                sx1, sy1, sx2, sy2 = map(int, sb.xyxy[0])
                sb_conf = float(sb.conf[0])
                label_name = sb_results.names[int(sb.cls[0])].lower()
                has_belt = "no" not in label_name and "without" not in label_name

                color = self.CLR_SEATBELT if has_belt else self.CLR_NO_SEATBELT
                tag = f"{'SEATBELT' if has_belt else 'NO SEATBELT'} {sb_conf:.2f}"

                # Map back to original frame coordinates
                abs_x1 = rx1 + sx1
                abs_y1 = ry1 + sy1
                abs_x2 = rx1 + sx2
                abs_y2 = ry1 + sy2

                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)
                self._label(frame, tag, abs_x1, abs_y1, color)

                if has_belt:
                    stats["seatbelt"] += 1
                else:
                    stats["no_seatbelt"] += 1

            # 4) Plate detection on ROI
            pl_results = self.plate_model(roi, conf=self.plate_conf, verbose=False)[0]
            for pl in pl_results.boxes:
                px1, py1, px2, py2 = map(int, pl.xyxy[0])
                pl_conf = float(pl.conf[0])

                abs_x1 = rx1 + px1
                abs_y1 = ry1 + py1
                abs_x2 = rx1 + px2
                abs_y2 = ry1 + py2

                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), self.CLR_PLATE, 2)
                self._label(frame, f"PLATE {pl_conf:.2f}", abs_x1, abs_y1, self.CLR_PLATE)
                stats["plates"] += 1

        return frame, stats
