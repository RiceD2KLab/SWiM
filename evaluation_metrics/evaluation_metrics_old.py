import argparse
import os
import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple
from skimage.metrics import hausdorff_distance

from ultralytics import YOLO
from src.YOLOv8Seg import YOLOv8Seg
import torch

class Evaluator():
    def __init__(self, model):
        """
        Initialize evaluator with the model

        Args:
            model (YOLOv8Seg model): YOLOv8Seg model in ONNX format 

        Returns:
            Object (Evaluator): Instance of Evaluation object
        """
        self.model = model

    def dice_coefficient(self, ground_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """
        Computes the Dice Coefficient between two binary masks.

        Args:
            ground_mask (numpy.ndarray): The first binary mask (H x W).
            pred_mask (numpy.ndarray): The second binary mask (H x W).

        Returns:
            float: The Dice Coefficients between the masks.  
        """
        intersect = np.sum(pred_mask*ground_mask)
        total_sum = np.sum(pred_mask) + np.sum(ground_mask)
        dice = np.mean(2*intersect/total_sum)
        return round(dice, 3) #round up to 3 decimal places
    
    def hausdorff_distance_contours(ground_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """
        Computes the Hausdorff Distance between the contours of two binary masks.

        Args:
            ground_mask (numpy.ndarray): The first binary mask (H x W).
            pred_mask (numpy.ndarray): The second binary mask (H x W).

        Returns:
            float: The Hausdorff Distance between the contours of the masks.
        """
        # Find contours of the masks
        contours1, _ = cv2.findContours(ground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours1 or not contours2:
            return 0.0  # If one of the masks is empty, return 0

        contour1 = contours1[0].squeeze()
        contour2 = contours2[0].squeeze()

        # Calculate Hausdorff distance
        hd = hausdorff_distance(contour1, contour2)
        return hd


    def binary_mask_to_yolo_poly(mask_path: str) -> Tuple[str]:
        """
        Convert a binary mask to a YOLO polygon string.

        Args:
            mask_path (str): path to the mask.

        Returns:
            Tuple[str]: List of YOLO polygon strings.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        size = mask.shape[::-1]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            # Skip small contours
            if len(cnt) < 3:
                continue
            cnt = cnt.squeeze() / size
            cnt = cnt.flatten().tolist()
            polygons.append(" ".join(map(str, cnt)))

        return polygons

    def evaluate(self, images_dir, txt_dir, preds_dir='./', metrics='all'):
        """
        Evaluates the model on specified metrics

        Args:
            images_dir: Path to the folder containing input images.
            txt_dir: Path to the folder containing ground truth masks.
            pred_dir: Path to the folder where predictions should be saved.
            metrics: Metrics to be calculated. Values can be one of 'dice', 'hausd' or 'all'.
        """

        # Validate metrics and set flags
        get_dice = False
        get_hausd = False

        if metrics == 'all':
            get_dice = True
            get_hausd = True
        elif metrics == 'dice':
            get_dice = True
        elif metrics == 'hausd':
            get_hausd = True
        else:
            # Throw error
            raise ValueError("Please enter a valid metric string from: 'dice', 'hausd' or 'all'.")

        # Load input images
        imgs_list = []
        results = []
        instance_dice_coefficients = []
        instance_hausdorff_distances = []
        if os.path.isdir(images_dir):
            with os.scandir(images_dir) as imgs_dir:
                for file in imgs_dir:
                    if file.is_file() and file.name.endswith('.png'):
                        #imgs_list.append(file.path)
                        img = cv2.imread(file.path)
                        result = self.model(img, conf_threshold=0.4, iou_threshold=0.45)
                        txt_path = os.path.join(txt_dir, os.path.basename(file.path).split('.')[0] + '.txt')
                        image, ground_truth_mask = self.binary_mask_to_yolo_poly(result.segments)
                        prediction_mask = eval.yolo_predictions_to_masks([result], img.shape)

                        if get_dice:
                            dice_coeff = eval.dice_coefficient(ground_truth_mask, prediction_mask)
                            instance_dice_coefficients.append(dice_coeff)
                        if get_hausd:
                            hd = eval.hausdorff_distance_contours(ground_truth_mask, prediction_mask)
                            instance_hausdorff_distances.append(hd)
                        
                        results.append(result)
        else: 
            #imgs_list.append(args.input)
            result = self.model(images_dir, conf_threshold=0.4, iou_threshold=0.45)
            results.append(result)

    

        
        