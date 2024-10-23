from ultralytics import YOLO

# Load a model
model = YOLO(
    "C:/Users/nikhi/Documents/Projects/NASA_segmentation_F24/experiment-runs/baseline-run2/weights/best.pt"
)

# Export the model
model.export(format="onnx")
