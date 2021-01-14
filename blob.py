import glob
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
import os
import shutil
from tqdm import tqdm


TRAIN = "train"
TEST = "test"
BALL = "ball"
NOT_BALL = "not_ball"


class BlobExtractor:
    def __init__(
        self, frames_path, xml_path, training_set, test_set, output_path, excludes
    ):
        self.frames_path = frames_path
        self.xml_path = xml_path
        self.training_set = training_set
        self.test_set = test_set
        self.output_path = output_path
        self.excludes = excludes

    def extract(self):
        try:
            shutil.rmtree(self.output_path)
        except FileNotFoundError:
            pass
        for x in (TRAIN, TEST):
            for y in (BALL, NOT_BALL):
                os.makedirs(str(Path(self.output_path) / x / y))
        for frame_set in glob.glob(str(Path(self.frames_path) / "*")):
            print(frame_set)
            self.extract_blobs_single_vid(frame_set)

    def extract_blobs_single_vid(self, vid_path):
        bg = cv2.createBackgroundSubtractorKNN()
        ball_i = not_ball_i = 0
        warned = False
        for i, frame_name in enumerate(
            tqdm(sorted(glob.glob(str(Path(vid_path) / "*"))))
        ):
            frame = cv2.imread(frame_name)
            assert frame is not None
            fg = bg.apply(frame)

            # skip first couple frames
            if i < 30:
                continue

            annotation_file = self.get_xml_file(frame_name)

            try:
                root = ET.parse(annotation_file)
                boxes = []
                for b in root.findall("object/bndbox"):
                    xmin = int(b.find("xmin").text)
                    xmax = int(b.find("xmax").text)
                    ymin = int(b.find("ymin").text)
                    ymax = int(b.find("ymax").text)
                    boxes.append((xmin, ymin, xmax, ymax))

            except FileNotFoundError as e:
                if not warned:
                    warned = True
                    print(f"WARNING: File not found - {annotation_file}")
                continue

            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [cnt for cnt in cnts if may_be_ball(cnt)]
            fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            fg = fg & frame

            parent = Path(frame_name).parts[-2]
            exclude = self.excludes.get(parent, {})
            balls, not_balls = bin_contours(cnts, boxes, exclude)

            is_test = annotation_file in self.test_set
            assert is_test or annotation_file in self.training_set
            for cnt in balls:
                ball_i += 1
                assert cv2.imwrite(
                    str(
                        Path(self.output_path)
                        / ["train", "test"][is_test]
                        / "ball"
                        / f"{parent}_{ball_i:08d}.png"
                    ),
                    cutout(cnt, frame),
                )

            for cnt in not_balls:
                not_ball_i += 1
                assert cv2.imwrite(
                    str(
                        Path(self.output_path)
                        / ["train", "test"][is_test]
                        / "not_ball"
                        / f"{parent}_{not_ball_i:08d}.png"
                    ),
                    cutout(cnt, frame),
                )

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


def bin_contours(cnts, boxes, exclude=None):
    """Return (ball, not_ball)"""
    ball = []
    not_ball = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        b = (x, y, x + w, y + h)
        ou = max(iou(b, box) for box in boxes) if boxes else 0.0
        if ou:
            ball.append(cnt)
        elif not (exclude and iou(exclude, b)):
            not_ball.append(cnt)
    return ball, not_ball
