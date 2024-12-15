import argparse
import torch
from ultralytics import YOLO

def main(args):
    # Load the YOLO model
    model = YOLO(args.yolo_path)

    # Create a random input tensor with the specified shape
    x = torch.randn((args.batch_size, 3, args.height, args.width), requires_grad=True)

    # Move the model and input to CUDA if specified
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x = x.to(device)

    # Profile the model execution
    with torch.autograd.profiler.profile(use_cuda=args.cuda) as prof:
        model(x)

    # Print the profiling results
    print(prof)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile YOLO model execution')
    parser.add_argument("--yolo-path", required=True, help="Path to the YOLO model file")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of the input tensor")
    parser.add_argument("--height", type=int, default=1024, help="Height of the input tensor")
    parser.add_argument("--width", type=int, default=1280, help="Width of the input tensor")
    parser.add_argument("--cuda", action='store_true', help="Use CUDA for profiling")
    args = parser.parse_args()
    main(args)