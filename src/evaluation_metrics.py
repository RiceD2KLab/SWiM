import argparse
import os
import cv2
import numpy as np
from skimage.draw import polygon
from skimage.metrics import hausdorff_distance
from skimage import metrics
from skimage.measure import find_contours
from scipy.spatial.distance import directed_hausdorff
from ultralytics import YOLO

def create_mask_from_polygons(image_shape, polygons):
    """Create a binary mask from a list of polygons."""
    mask = np.zeros(image_shape, dtype=np.uint8)  # Create a black mask
    for poly in polygons:
        # Convert normalized coordinates to pixel values
        poly = np.array(poly) * np.array(image_shape[::-1])  # Reverse shape for (height, width)
        rr, cc = polygon(poly[:, 1], poly[:, 0], mask.shape)  # Note: y,x order for polygon
        mask[rr, cc] = 1  # Fill the polygon area with white (1)
    return mask

def process_yolo_txt_file(txt_file_path, image_shape):
    """Process a YOLO format TXT file and create masks."""
    polygons = []
    n_masks = 0
    with open(txt_file_path, 'r') as file:
        for line in file:
            if n_masks > 1:
                # if there are more than 1 spacecraft in the image, skip it.
                return None
            
            parts = list(map(float, line.strip().split()))
            class_index = int(parts[0])  # First part is the class index
            coords = parts[1:]  # Remaining parts are the coordinates
            
            # Group coordinates into pairs (x, y)
            poly_points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(poly_points)
            
            n_masks += 1

    # Create the mask from polygons
    mask_image = create_mask_from_polygons(image_shape, polygons)
    
    return mask_image

def get_pred_and_true(image_path, txt_file_path, model, og_shape=(1024, 1280)):
    """Get predicted and true masks."""
    prediction = model(image_path)
    y_pred = (prediction[0].masks.data.squeeze(0) * 255).numpy().astype(np.uint8)
    
    mask_image = process_yolo_txt_file(txt_file_path, og_shape)
    y_true = cv2.resize(mask_image * 255, (y_pred.shape[1], y_pred.shape[0]))

    return y_true, y_pred

def dice(y_true, y_pred, k=255):
    """Calculate Dice coefficient."""
    intersection = np.sum(y_pred[y_true == k]) * 2.0
    dice_score = intersection / (np.sum(y_pred) + np.sum(y_true))
    return dice_score

def hausdorff_distance_mask(y_true, y_pred):
    """Calculate the Hausdorff distance between two segmentation masks."""
    #return max(directed_hausdorff(y_pred, y_true)[0], directed_hausdorff(y_true, y_pred)[0])
    return hausdorff_distance(y_true, y_pred, 'modified')

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks using YOLO.")
    
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images.")
    
    parser.add_argument("--txt_dir", type=str, required=True,
                        help="Directory containing YOLO format TXT files.")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained YOLO model.")
    
    parser.add_argument("--log_dir", type=str, default=os.path.expanduser("~/"), 
                        help="Directory to save log file (default: root directory).")
    
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.model_path)

    # Prepare log file path
    log_file_path = os.path.join(args.log_dir, "metrics_log.txt")

    # Open the log file for writing metrics
    with open(log_file_path, 'w') as log_file:
        log_file.write("Image Name,Dice Coefficient,Hausdorff Distance\n")

        # Process each image in the specified directory
        for image_name in os.listdir(args.image_dir):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
                image_path = os.path.join(args.image_dir, image_name)
                txt_file_path = os.path.join(args.txt_dir, os.path.splitext(image_name)[0] + '.txt')
                
                if not os.path.exists(txt_file_path):
                    print(f"Warning: No corresponding TXT file found for {image_name}. Skipping.")
                    continue
                
                img = cv2.imread(image_path)
                y_true, y_pred = get_pred_and_true(image_path, txt_file_path, model)
                print('Shapes of img, y_true, y_pred', img.shape, y_true.shape, y_pred.shape)
                
                # Calculate metrics
                
                dice_score = dice(y_true, y_pred)
                hausdorff_dist = hausdorff_distance_mask(y_true, y_pred)

                # Log metrics to file
                log_file.write(f"{image_name},{dice_score:.4f},{hausdorff_dist:.4f}\n")
                print(f"Processed {image_name}: Dice={dice_score:.4f}, Hausdorff={hausdorff_dist:.4f}")

if __name__ == "__main__":
    main()