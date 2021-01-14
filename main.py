import argparse
import json

from blob import BlobExtractor


def main():
    parser = argparse.ArgumentParser(description="Track badminton shuttles")
    parser.add_argument(
        "--frames_path",
        help="Directory containing collections of frames",
        default="data/frames",
    )
    parser.add_argument(
        "--xml_path",
        help="Directory containing collections of annotation files",
        default="data/xml",
    )
    parser.add_argument(
        "--training_set",
        help="Subset of the data for training. newline delimited filenames with relative root FRAMES_PATH",
        default="data/train.txt",
    )
    parser.add_argument(
        "--test_set",
        help="Subset of the data for testing. newline delimited filenames with relative root FRAMES_PATH",
        default="data/test.txt",
    )
    parser.add_argument(
        "--blob_path", help="Directory for blob cutouts", default="data/blobs"
    )

    parser.add_argument(
        "--excludes",
        help="Exclusion JSON config file",
        default="data/exclude.json",
    )
    args = parser.parse_args()
    with open(args.training_set) as f:
        training_set = set(f.read().splitlines())
    with open(args.test_set) as f:
        test_set = set(f.read().splitlines())
    try:
        with open(args.excludes) as f:
            excludes = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {args.excludes} not found")
        excludes = {}

    extractor = BlobExtractor(
        args.frames_path, args.xml_path, training_set, test_set, args.blob_path, excludes
    )
    extractor.extract()

    # train_classifier()
    # infer()
    # evaluate()


if __name__ == "__main__":
    main()