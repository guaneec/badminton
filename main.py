import argparse
import json

from blob import BlobExtractor

steps = {k: v for v, k in enumerate(["blob", "train", "infer", "eval"])}


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

    parser.add_argument(
        "--start_from",
        help="Step to start from (blob -> train -> infer -> eval",
        choices=steps.keys(),
        default="blob",
    )

    parser.add_argument(
        "--model_path", help="Path to store the trained model", default="data/model"
    )

    parser.add_argument(
        "--prediction_path",
        help="Path to store the prediction results",
        default="data/prediction.json",
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

    if steps[args.start_from] <= steps["blob"]:
        extractor = BlobExtractor(
            args.frames_path,
            args.xml_path,
            training_set,
            test_set,
            args.blob_path,
            excludes,
        )
        extractor.extract()

    if steps[args.start_from] <= steps["train"]:
        from train import train_model

        model = train_model(args.blob_path)
        model.save(args.model_path)
    else:
        import tensorflow as tf
        from misc import fix_conv
        fix_conv()
        model = tf.keras.models.load_model(args.model_path)

    if steps[args.start_from] <= steps["infer"]:
        from infer import Predictor
        predictor = Predictor(model, args.frames_path, args.xml_path)
        preds = predictor.predict(training_set)
        with open(args.prediction_path, "w") as f:
            json.dump(preds, f)

    # evaluate()


if __name__ == "__main__":
    main()