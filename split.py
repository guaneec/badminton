import argparse
import glob
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--xml_path",
    help="Directory containing collections of annotation files",
    default="data/xml",
)
parser.add_argument(
    "--training_set",
    help="Training set output file",
    default="data/train.txt",
)
parser.add_argument(
    "--test_set",
    help="Test set output file",
    default="data/test.txt",
)

args = parser.parse_args()

filenames = glob.glob(str(Path(args.xml_path) / "*" / '*'))

n = len(filenames)
TEST_SPLIT = 0.2

n_test = int(TEST_SPLIT * n)
n_train = n - TEST_SPLIT

assert n_test > 0 and n_train > 0

random.shuffle(filenames)

with open(args.test_set, "w") as f:
    for filename in filenames[:n_test]:
        print(filename, file=f)

with open(args.training_set, "w") as f:
    for filename in filenames[n_test:]:
        print(filename, file=f)