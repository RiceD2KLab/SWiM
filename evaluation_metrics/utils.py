import argparse
import os
import cv2
import numpy as np
from skimage.metrics import hausdorff_distance

def load_instance_segmentation_data(image_path, txt_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    ground_truth_masks = []
    for line in lines:
        values = line.strip().split()
        label = int(values[0])
        polygon_coords = list(map(int, values[1:]))
        polygon_coords = np.array(polygon_coords).reshape(-1, 2)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [polygon_coords], -1, 255, -1)
        ground_truth_masks.append(mask)

    return image, ground_truth_masks

def make_yolo_predictions(model, image):
    # Assuming you have a function to preprocess the image and make predictions
    predictions = model.predict(image)
    return predictions

def yolo_predictions_to_masks(predictions, image_shape):
    height, width, _ = image_shape
    prediction_masks = []

    for prediction in predictions:
        mask = np.zeros((height, width), dtype=np.uint8)
        for pred in prediction:
            x, y, w, h, _, _ = pred
            x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
            x2, y2 = int((x + w / 2) * width), int((y + h / 2) * height)
            # Assuming polygon coordinates are available in the prediction
            # For simplicity, using bounding box coordinates here
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        prediction_masks.append(mask)

    return prediction_masks

def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def hausdorff_distance_contours(mask1, mask2):
    # Find contours of the masks
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours1 or not contours2:
        return 0.0  # If one of the masks is empty, return 0

    contour1 = contours1[0].squeeze()
    contour2 = contours2[0].squeeze()

    # Calculate Hausdorff distance
    hd = hausdorff_distance(contour1, contour2)
    return hd

def main():
    parser = argparse.ArgumentParser(description='Calculate Dice Coefficient and Hausdorff Distance for Instance Segmentation')
    parser.add_argument('--image_dir', help='Path to the directory containing images', required=True)
    parser.add_argument('--txt_dir', help='Path to the directory containing TXT files', required=True)
    parser.add_argument('--model_path', help='Path to the YOLO model', required=True)

    args = parser.parse_args()

    image_dir = args.image_dir
    txt_dir = args.txt_dir
    model_path = args.model_path

    # Initialize YOLO model
    # Assuming you have a function to load the YOLO model
    yolo_model = # Load your YOLO model here

    dice_coefficients = []
    hausdorff_distances = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            txt_path = os.path.join(txt_dir, filename.split('.')[0] + '.txt')

            image, ground_truth_masks = load_instance_segmentation_data(image_path, txt_path)
            predictions = make_yolo_predictions(yolo_model, image)
            prediction_masks = yolo_predictions_to_masks(predictions, image.shape)

            # Calculate Dice coefficient for each instance
            instance_dice_coefficients = []
            instance_hausdorff_distances = []
            for ground_truth_mask, prediction_mask in zip(ground_truth_masks, prediction_masks):
                dice_coeff = dice_coefficient(ground_truth_mask, prediction_mask)
                instance_dice_coefficients.append(dice_coeff)

                hd = hausdorff_distance_contours(ground_truth_mask, prediction_mask)
                instance_hausdorff_distances.append(hd)

            # Average Dice coefficient and Hausdorff distance for all instances in the image
            average_dice_coeff = np.mean(instance_dice_coefficients)
            average_hausdorff_distance = np.mean(instance_hausdorff_distances)

            dice_coefficients.append(average_dice_coeff)
            hausdorff_distances.append(average_hausdorff_distance)

    # Print the average Dice coefficient and Hausdorff distance across all images
    average_dice_coefficient = np.mean(dice_coefficients)
    average_hausdorff_distance = np.mean(hausdorff_distances)
    print(f"Average Dice Coefficient: {average_dice_coefficient}")
    print(f"Average Hausdorff Distance: {average_hausdorff_distance}")

if __name__ == "__main__":
    main()