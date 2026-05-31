"""
detector.py — framework-independent inference for the cat detector.

Loads an exported YOLO26 ONNX model once and exposes `predict(image_path)`
returning bounding boxes in ORIGINAL-image pixel coordinates.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf: float = 0.25,  # Increased from baseline to filter low-conf noise on holdout
        iou: float = 0.7,
        class_names=("cat",),
    ):
        # Force single-threaded CPU evaluation context for stable, isolated leaderboard scoring
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resizes and pads an image into a square canvas.
        Strictly synchronized with the notebook's round-to-nearest pixel math.
        """
        w0, h0 = img.size
        r = min(self.imgsz / h0, self.imgsz / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        dw, dh = (self.imgsz - nw) / 2, (self.imgsz - nh) / 2
        
        if (nw, nh) != (w0, h0):
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            
        left, top = round(dw - 0.1), round(dh - 0.1)
        
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(img, (left, top))
        
        # Normalize and change channel layout from HWC to CHW -> (1, 3, H, W)
        x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), r, (left, top)

    def predict(self, image_path: str) -> list[dict]:
        """Runs inference and decodes bounding boxes back to original-image pixel space."""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return []  # Gracefully catch corrupted frames during holdout

        W, H = img.size
        x, r, (px, py) = self._letterbox(img)
        
        # Execute ONNX session graph
        outputs = self.session.run(None, {self.input_name: x})[0]
        
        # Determine head variant based on shape dimensions
        if len(outputs.shape) == 3 and outputs.shape[2] == 6:
            # End-to-End Head: Shape (1, 300, 6) -> [x1, y1, x2, y2, score, class]
            detections = outputs[0]
        else:
            # Legacy One-to-Many Head Fallback: Shape (1, 5, 8400) -> Needs Manual NMS
            detections = self._decode_legacy(outputs[0])

        results = []
        for x1, y1, x2, y2, score, cls in detections:
            if score < self.conf:
                continue
            
            # Reverse letterbox transformations back to true image space
            real_x1 = max(0.0, min(W, (x1 - px) / r))
            real_y1 = max(0.0, min(H, (y1 - py) / r))
            real_x2 = max(0.0, min(W, (x2 - px) / r))
            real_y2 = max(0.0, min(H, (y2 - py) / r))
            
            cls_idx = int(cls)
            cls_str = self.class_names[cls_idx] if cls_idx < len(self.class_names) else "cat"

            results.append({
                "xmin": float(real_x1),
                "ymin": float(real_y1),
                "xmax": float(real_x2),
                "ymax": float(real_y2),
                "confidence": float(score),
                "class": cls_str,
            })
            
        return sorted(results, key=lambda b: -b["confidence"])

    def _decode_legacy(self, pred: np.ndarray) -> np.ndarray:
        """Fallback anchor decoding method for legacy one-to-many heads."""
        # Input shape: (4 + num_classes, 8400) -> Transpose to (8400, 5)
        pred = pred.T  
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        
        cls = scores_all.argmax(axis=1)
        score = scores_all.max(axis=1)

        keep = score >= self.conf
        boxes_xywh, score, cls = boxes_xywh[keep], score[keep], cls[keep]
        if boxes_xywh.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        cx, cy, w, h = boxes_xywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        kept = self._nms(xyxy, score, self.iou)
        return np.concatenate(
            [xyxy[kept], score[kept, None], cls[kept, None].astype(np.float32)],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
        """Pure NumPy vectorized NMS for legacy parsing."""
        x1, y1, x2, y2 = boxes.T
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
            
        return keep