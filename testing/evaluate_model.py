import argparse
import os

from testing.YOLOv8Seg import YOLOv8Seg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks using YOLO."
    )

    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images."
    )

    parser.add_argument(
        "--txt_dir",
        type=str,
        required=True,
        help="Directory containing YOLO format TXT files.",
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained YOLO model."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.expanduser("~/"),
        help="Directory to save log file (default: root directory).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Metrics on which to evaluate the model. Select one of 'dice', 'hausd' and 'all'.",
    )
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLOv8Seg(args.model_path)

    # Evaluate model
    model.evaluate(args.image_dir, args.txt_dir, args.log_dir, metrics="all")


if __name__ == "__main__":
    main()
