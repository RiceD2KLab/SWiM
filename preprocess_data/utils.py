import cv2
import os
import numpy as np
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (128, 128, 128) # Gray
    ]

def mask_to_polygons_multi(mask):
    h, w = mask.shape
    polygons = []
    labels = []
    annotations = []

    for label in np.unique(mask):
        if label == 0:
            continue  # Skip background

        binary_mask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the contour has at least 4 points
            if len(approx) < 4:
                continue  # Skip this contour if it has fewer than 4 points
            
            try:
                poly = Polygon(approx.reshape(-1, 2))
                poly = poly.simplify(1, preserve_topology=True)
                
                if poly.is_valid and poly.area > 0:
                    min_x, min_y, max_x, max_y = poly.bounds
                    
                    coords = np.array(poly.exterior.coords)
                    coords[:, 0] /= w
                    coords[:, 1] /= h
                    
                    polygons.append(coords.flatten().tolist())
                    labels.append(int(label))

                    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                    
                    annotations.append({
                        "segmentation": [coords.flatten().tolist()],
                        "category_id": int(label),
                        "bbox": bbox,
                        "area": poly.area
                    })
            except Exception as e:
                print(f"Error processing contour: {e}")
                continue  # Skip this contour if there's any error

    return polygons, labels, annotations

def annotations_to_mask(annotations, image_shape=(256,256)):
    """Convert COCO-style annotations back to a mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for ann in annotations:
        polygon = np.array(ann['segmentation'][0]).reshape(-1, 2)
        polygon = (polygon * np.array([image_shape[1], image_shape[0]])).astype(int)
        cv2.fillPoly(mask, [polygon], int(ann['category_id'] + 1))
    return mask


def overlay_mask_on_image_with_boxes(image, mask, annotations, alpha=0.5):
    """Overlay a colored mask on an image with bounding boxes and labels."""
    # Convert grayscale image to RGB
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image

    # Define a color map with distinct colors for each class
    
    
    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(1, mask.max() + 1):
        colored_mask[mask == i] = colors[i % len(colors)]

    # Overlay the mask on the image
    overlay = (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_image)

    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        color = colors[(ann['category_id']) % len(colors)]
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)

    return np.array(overlay_image)

def extract_instance(image, mask, instance_id):
    """Extract a single instance from the image using the mask."""
    instance_mask = (mask == instance_id).astype(np.uint8) * instance_id
    roi = cv2.bitwise_and(image, image, mask=(instance_mask > 0).astype(np.uint8))
    return roi, instance_mask

def paste_instance(base_image, roi, instance_mask):
    """Paste the extracted ROI onto a copy of the base image."""
    result_image = base_image.copy()
    result_image[instance_mask > 0] = roi[instance_mask > 0]
    return result_image

def process_mask_images_to_poygon_txt(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif'))]

    for mask_file in mask_files:
        mask_path = os.path.join(input_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        polygons, labels, _ = mask_to_polygons_multi(mask)
        
        txt_filename = os.path.splitext(mask_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for polygon, label in zip(polygons, labels):
                # Subtract 1 from label to start class indexing from 0 for YOLO format
                line = f"{label-1} " + " ".join(map(str, polygon))
                f.write(line + '\n')

def process_tif_to_png_and_verify(tif_path):
    # Load the TIFF image using cv2
    original_image = cv2.imread(tif_path)
    
    if original_image is None:
        raise ValueError(f"Failed to load image: {tif_path}")
    
    # Construct the output PNG path
    png_path = tif_path.replace('.tif', '.png')
    
    # Save the image as PNG
    cv2.imwrite(png_path, original_image)
    
    # Read back the PNG image
    loaded_image = cv2.imread(png_path)
    
    # Compare the arrays
    are_equal = np.array_equal(original_image, loaded_image)
    
    # Delete the PNG file
    os.remove(png_path)
    
    return are_equal, original_image, loaded_image


def create_dataset_df(base_dir):
    data = []
    label_dir = os.path.join(base_dir, 'labels')
    image_dir = os.path.join(base_dir, 'preprocessed_images')
    
    # Iterate through all txt files in the label directory
    for txt_file in Path(label_dir).glob('*.txt'):
        txt_path = str(txt_file.relative_to(base_dir))
        png_name = txt_file.stem + '.png'
        png_path = str(Path('preprocessed_images') / png_name)
        
        # Check if the corresponding image file exists
        if not os.path.exists(os.path.join(base_dir, png_path)):
            print(f"Warning: Image file not found for {txt_path}")
            continue
        
        # Read labels from the txt file
        with open(txt_file, 'r') as f:
            for line in f:
                label = line.strip().split()[0]  # Get the first value of each line
                data.append([txt_path, png_path, label])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['label_file', 'image_file', 'label'])
    return df

def split_dataset(df):
    # Create unique image-label pairs
    unique_pairs = df[['label_file', 'image_file', 'label']].drop_duplicates().reset_index(drop=True)
    
    # Split into train and test (85:15)
    train_val, test = train_test_split(unique_pairs, test_size=0.05, stratify=unique_pairs['label'], random_state=42)
    
    # Further split train into train and validation (80:20 of the original train set)
    train, val = train_test_split(train_val, test_size=0.15, stratify=train_val['label'], random_state=42)
    
    return train, val, test

def move_files(df, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for _, row in df.iterrows():
        image_file = os.path.join(source_dir, row['image_file'])
        label_file = os.path.join(source_dir, row['label_file'])
        
        shutil.copy(image_file, dest_dir)
        shutil.copy(label_file, dest_dir)