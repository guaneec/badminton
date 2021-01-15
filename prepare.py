import argparse
from pathlib import Path
import glob
import shutil

parser = argparse.ArgumentParser()

parser.add_argument(
    "--src", help="Root of the extracted BadmintonData.zip (cleaned)", required=True
)
parser.add_argument(
    "--dst_xml", help="Destination of the annotation files", default="data/xml"
)
parser.add_argument(
    "--dst_frames", help="Destination of the image files", default="data/frames"
)

parser.add_argument(
    "--include", help="Comma delimited group ids to use, e.g. '02,17,18'", default=""
)
parser.add_argument(
    "--exclude",
    help="Comma delimited group ids to NOT use, e.g. '02,17,18'",
    default="",
)

args = parser.parse_args()

includes = [x for x in args.include.split(",") if x]
excludes = [x for x in args.exclude.split(",") if x]
if includes and excludes:
    raise RuntimeError("Include and Exclude should not be both specified")


for d in (args.dst_xml, args.dst_frames):
    try:
        shutil.rmtree(d)
    except FileNotFoundError:
        pass

for g in sorted(glob.glob(str(Path(args.src) / "group*"))):
    gid = g[-2:]
    if includes and gid not in includes or gid in excludes:
        continue
    print(gid)
    shutil.copytree(str(Path(g) / "annotation"), str(Path(args.dst_xml) / gid))
    shutil.copytree(str(Path(g) / "image"), str(Path(args.dst_frames) / gid))