import argparse
import os
import cv2
import numpy as np
from skimage.metrics import hausdorff_distance

from ultralytics import YOLO
import torch

class Evaluator():
    def __init__(self, model_path, images_dir, txt_dir, preds_dir):
        self.model = model_path
        self.images_dir = images_dir
        self.txt_dir = txt_dir
        self.pred_dir = preds_dir

    def dice_coefficient(self, ground_mask, pred_mask):
        intersect = np.sum(pred_mask*ground_mask)
        total_sum = np.sum(pred_mask) + np.sum(ground_mask)
        dice = np.mean(2*intersect/total_sum)
        return round(dice, 3) #round up to 3 decimal places
    
    def hausdorff_distance_contours(mask1, mask2):
        """
        Computes the Hausdorff Distance between the contours of two binary masks.

        Args:
            mask1 (numpy.ndarray): The first binary mask (H x W).
            mask2 (numpy.ndarray): The second binary mask (H x W).

        Returns:
            float: The Hausdorff Distance between the contours of the masks.
        """
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

    