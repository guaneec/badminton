import argparse
import glob
from tqdm import tqdm
from pathlib import Path
import os
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("root", help="Root of the extracted BadmintonData.zip")
args = parser.parse_args()


def filter_non_fhd():
    # last frame of each group, 0 if N/A
    last = {
        "01": 0,
        "02": 0,
        "03": 0,
        "04": 1248,
        "05": 1421,
        "06": 1275,
        "07": 1405,
        "08": 1323,
        "09": 1350,
        "10": 1314,
        "11": 0,
        "13": 1659,
        "14": 1606,
        "15": 1463,
        "16": 1391,
        "17": 1593,
        "18": 1487,
        "19": 0,
        "20": 0,
        "21": 0,
        "22": 1638,
        "23": 0,
    }

    for k, v in last.items():
        print(k)
        for sub in ("annotation", "image"):
            print(sub)
            for f in tqdm(glob.glob(str(Path(args.root) / f"group{k}/{sub}/*"))):
                i = int(Path(f).stem)
                if v and i > v:
                    os.unlink(f)
                    print("Removed", f)


def clean_boxes():
    # Remove zero-sized boxes and fix typos
    for f in tqdm(glob.glob(str(Path(args.root) / "*" / "annotation" / "*"))):
        tree = ET.parse(f)
        root = tree.getroot()
        modified = False
        for o in root.findall("object"):
            if o.find("name").text != "badminton":
                o.find("name").text = "badminton"
                modified = True
            xmin = int(o.find("bndbox/xmin").text)
            xmax = int(o.find("bndbox/xmax").text)
            ymin = int(o.find("bndbox/ymin").text)
            ymax = int(o.find("bndbox/ymax").text)
            if xmin >= xmax or ymin >= ymax:
                root.remove(o)
                modified = True
        if not modified:
            continue
        if not root.findall("object"):
            os.unlink(f)
            print("Removed", f)
        else:
            tree.write(f)
            print("Rewritten", f)


filter_non_fhd()
clean_boxes()