from pathlib import Path
import glob
import cv2
from misc import read_boxes
from tqdm import tqdm
import tensorflow as tf


class Predictor:
    def __init__(self, model, frames_path, xml_path):
        self.model = model
        self.frames_path = frames_path
        self.xml_path = xml_path

    def predict(self, subset):
        """return list of (iou, confidence) sorted by descending confidence"""
        preds = []
        for frame_set in glob.glob(str(Path(self.frames_path) / "*")):
            print(frame_set)
            preds.extend(self._predict_single_vid(frame_set, subset))
        preds = sorted(preds, key=lambda x: -x[1])
        return preds

    def _predict_single_vid(self, vid_path, subset):
        bg = cv2.createBackgroundSubtractorKNN()
        for i, frame_name in enumerate(
            tqdm(sorted(glob.glob(str(Path(vid_path) / "*"))))
        ):
            frame = cv2.imread(frame_name)
            assert frame is not None
            bg.apply(frame)
            fg = bg.apply(frame)

            # skip first couple frames
            if i < 30:
                continue

            annotation_file = self.get_xml_file(frame_name)
            if annotation_file not in subset:
                continue

            boxes = read_boxes(annotation_file)

            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [cnt for cnt in cnts if may_be_ball(cnt)]
            fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            fg = fg & frame

            for cnt in cnts:
                patch = cutout(cnt, frame)
                patch = tf.image.resize(patch, (32, 32))
                p = float(self.model.predict(patch[None]).ravel()[0])
                x, y, w, h = cv2.boundingRect(cnt)
                b = (x, y, x + w, y + h)
                o = max(iou(b, box) for box in boxes) if boxes else 0.0
                yield o, p, frame_name, b

    def get_xml_file(self, frame_file):
        p = Path(frame_file)
        return str(Path(self.xml_path) / p.parts[-2] / f"{p.stem}.xml")


def may_be_ball(contour):
    area = cv2.contourArea(contour)
    if not 10 <= area <= 500:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if not area / hull_area > 0.5:
        return False
    return True


def cutout(contour, im):
    x, y, w, h = cv2.boundingRect(contour)
    return im[y : y + h, x : x + w]


def iou(r1, r2):
    x11, y11, x21, y21 = r1
    x12, y12, x22, y22 = r2
    lx = max(x11, x12)
    rx = min(x21, x22)
    ty = max(y11, y12)
    by = min(y21, y22)
    if lx >= rx or ty >= by:
        return 0.0
    i = (rx - lx) * (by - ty)
    return i / ((x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 - y12) - i)