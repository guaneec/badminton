import json
import argparse
import cv2
from collections import defaultdict
from tqdm import tqdm
import glob
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prediction_path",
    help="Path to prediction results",
    default="data/prediction.json",
)

parser.add_argument("--vid_path", help="Path to frames to visualize", required=True)

args = parser.parse_args()

print("loading predictions")
with open(args.prediction_path) as f:
    preds = json.load(f)

grouped_preds = defaultdict(list)

for p in tqdm(preds):
    _, _, f, _, _ = p
    grouped_preds[f].append(p)

for f in sorted(glob.glob(str(Path(args.vid_path) / "*"))):
    print(f, len(grouped_preds[f]))
    im = cv2.imread(f)
    for p in grouped_preds[f]:
        x1, y1, x2, y2 = p[3]
        if p[1] < 0.5:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0))
        else:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("viz", im)
    k = cv2.waitKey(16)

    if k == ord("p"):
        k = cv2.waitKey()
    if k == ord("q"):
        break

cv2.destroyAllWindows()