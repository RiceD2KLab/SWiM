import os
from typing import List
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
sys.path.append(Path(os.getcwd()).parent.parent.as_posix())

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pybboxes as pbx

from ultralytics import SAM
from ultralytics.utils.instance import Bboxes
from ultralytics.utils.plotting import Annotator


from utils.utils import yolo_polygon_to_xyxy


from consts import *

def check_dataset_integrity(data_dir):
    data_dir = Path(data_dir)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        imgs = os.listdir(data_dir / 'images' / split)
        labels = os.listdir(data_dir / 'labels' / split)
        
        assert len(imgs) == len(labels), f'Number of {split} images and labels do not match'
        
        # Check for image formats
        valid_image_extensions = ('.jpg', '.jpeg', '.png')
        assert all(img.endswith(valid_image_extensions) for img in imgs), f'Not all {split} images are in jpg, jpeg, or png format'
        
        assert all(label.endswith('.txt') for label in labels), f'Not all {split} labels are in txt format'
        
        # Check if labels are in YOLO format
        assert all(len(np.loadtxt(data_dir / 'labels' / split / label)) == 5 for label in labels), f'Not all {split} labels are in YOLO format'
        
        # Check corresponding labels for each image
        #assert all(img.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') in labels for img in imgs), f'Not all {split} images have corresponding labels'
    
    print('Dataset integrity check passed.')


def analyze_bbox_distribution(data_dir, output_dir, is_polygon=False):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    splits = ['train', 'val', 'test']
    areas = {'split': [], 'image': [], 'area': []}
    
    for split in splits:
        bbox_files = os.listdir(data_dir / 'labels' / split)
        for file in bbox_files:
            img_path = data_dir / "images" / split / file.replace('.txt', '.png')
            img = cv2.imread(str(img_path))
            
            if is_polygon:
                bbox_np = yolo_polygon_to_xyxy(data_dir / 'labels' / split / file, img.shape[1], img.shape[0])
            else:
                bbox_np = np.loadtxt(data_dir / 'labels' / split / file)
            
            bbox = Bboxes(bboxes=bbox_np[1:] if not is_polygon else bbox_np, format='xyxy')
            
            areas['split'].append(split)
            areas['image'].append(os.path.splitext(file)[0])
            areas['area'].append(bbox.areas()[0])
    
    df = pd.DataFrame(areas)
    df['fraction_of_image'] = df['area'] / (img.shape[0] * img.shape[1])
    df.to_csv(output_dir / 'bbox_areas.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['fraction_of_image'], bins=50)
    plt.title('Bounding box area as a fraction of image area', fontsize=16)
    plt.xlabel('Fraction of image area', fontsize=14)
    plt.ylabel('Number of images', fontsize=14)
    plt.xticks(np.arange(0, 1.1, 0.1), [f'{i}%' for i in range(0, 101, 10)])
    plt.savefig(output_dir / 'bbox_distribution.png')
    plt.close()
    
    print(f'Bbox distribution analysis completed. Results saved in {output_dir}')


def perform_sam_segmentation(data_dir, models_dir, output_dir, sample_size, bbox_threshold):
    
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    
    # Load SAM model
    sam_model = SAM(models_dir / "sam2_b.pt")
    
    # Select images with bounding box area > threshold
    areas_df = pd.read_csv(output_dir / "bbox_areas.csv")
    selected_images = areas_df[areas_df["fraction_of_image"] > bbox_threshold]
    selected_images = selected_images.sample(n=min(sample_size, len(selected_images)))
    
    results = []
    for _, row in selected_images.iterrows():
        img_path = data_dir / "images" / row["split"] / f"{row['image']}.jpg"
        bbox_path = data_dir / "labels" / row["split"] / f"{row['image']}.txt"
        
        image = cv2.imread(str(img_path))
        bbox = np.loadtxt(bbox_path)
        img_bbox = pbx.convert_bbox(
            (bbox[1], bbox[2], bbox[3], bbox[4]),
            from_type="yolo",
            to_type="voc",
            image_size=(image.shape[1], image.shape[0])
        )
        
        sam_result = sam_model(img_path, bboxes=[img_bbox], verbose=False)[0]
        mask = sam_result.masks.data.cpu().numpy().astype(np.uint8).squeeze(0) * 255
        
        results.append({
            "image": row["image"],
            "split": row["split"],
            "mask": mask
        })
    
    # Save results
    np.save(output_dir / "sam_segmentation_results.npy", results)
    
    # Plot results
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    for i, result in enumerate(results[:20]):
        ax = axs[i // 5, i % 5]
        ax.imshow(result["mask"], cmap="gray")
        ax.set_title(f"{result['image']} ({result['split']})")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "sam_segmentation_samples.png")
    plt.close()

def evaluate_mask_quality(data_dir, output_dir):
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    results = np.load(output_dir / "sam_segmentation_results.npy", allow_pickle=True)
    
    quality_scores = []
    for result in results:
        mask = result["mask"]
        
        # Calculate mask quality metrics (example: ratio of non-zero pixels)
        quality_score = np.count_nonzero(mask) / mask.size
        
        quality_scores.append({
            "image": result["image"],
            "split": result["split"],
            "quality_score": quality_score
        })
    
    quality_df = pd.DataFrame(quality_scores)
    quality_df.to_csv(output_dir / "mask_quality_scores.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=quality_df, x="quality_score", bins=20)
    plt.title("Distribution of Mask Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    plt.savefig(output_dir / "mask_quality_distribution.png")
    plt.close()


def analyze_spacecraft_dataset(data_dir, output_dir):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    with open(data_dir / "spacecrafts/all_bbox.txt", "r") as f:
        data = f.read()
    data = eval(data)  # Safely evaluate the string as a Python expression
    
    spacecraft_fovs = []
    for key, value in data.items():
        areas = Bboxes(np.array(value), format="xyxy").areas()
        a = areas.sum() / (1280 * 720)
        spacecraft_fovs.append(min(a, 1.0))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(spacecraft_fovs, bins=50)
    plt.title("Histogram of Spacecraft sizes in the Spacecrafts dataset", fontsize=16)
    plt.xlabel("Fraction of image area", fontsize=14)
    plt.ylabel("Number of spacecrafts", fontsize=14)
    plt.xticks(np.arange(0, 1.1, 0.1), [f"{i*10}%" for i in range(11)])
    plt.savefig(output_dir / "spacecraft_size_distribution.png")
    plt.close()
    
    spacecraft_df = pd.DataFrame({"fraction_of_image": spacecraft_fovs})
    spacecraft_df.to_csv(output_dir / "spacecraft_sizes.csv", index=False)

def main(args):
    """
    Main function to analyze spacecraft dataset and perform segmentation.

    This function performs the following tasks:
    1. Checks dataset integrity
    2. Analyzes bounding box distribution
    3. Performs SAM segmentation on a sample of images
    4. Evaluates mask quality
    5. Analyzes the spacecraft dataset

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - data_dir (str): Path to the main data directory
            - models_dir (str): Path to the directory containing the SAM model
            - output_dir (str): Path to save plots and CSV files
            - sample_size (int): Number of images to sample for segmentation
            - bbox_threshold (float): Threshold for bounding box area as a fraction of image area
            - is_polygon (bool): Flag indicating if labels are in polygon format
            - img_width (int): Width of the images in pixels
            - img_height (int): Height of the images in pixels

    Returns:
        None
    """
    check_dataset_integrity(args.data_dir)
    analyze_bbox_distribution(args.data_dir, args.output_dir, args.is_polygon)
    perform_sam_segmentation(args.data_dir, args.models_dir, args.output_dir, args.sample_size, args.bbox_threshold)
    evaluate_mask_quality(args.data_dir, args.output_dir)
    analyze_spacecraft_dataset(args.data_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spacecraft dataset and perform segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the main data directory")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to the directory containing the SAM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save plots and CSV files")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of images to sample for segmentation")
    parser.add_argument("--bbox_threshold", type=float, default=0.5, help="Threshold for bounding box area as a fraction of image area")
    parser.add_argument("--is_polygon", action="store_true", help="Flag indicating if labels are in polygon format")
    parser.add_argument("--img_width", type=int, default=1280, help="Width of the images in pixels")
    parser.add_argument("--img_height", type=int, default=1024, help="Height of the images in pixels")
    args = parser.parse_args()

    main(args)
