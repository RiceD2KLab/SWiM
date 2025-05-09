# Overview
=====================================
The purpose of this repository is to provide scripts and functions for processing and converting image masks into polygon annotations, which are commonly used in object detection and segmentation tasks.

## Functions
-------------

### `mask_to_polygons_multi`
#### Description
This function converts a multi-class mask image into a list of polygons, labels, and annotations.

#### Parameters
- **mask**: A 2D numpy array representing the mask image.

#### Returns
- **polygons**: A list of polygon coordinates.
- **labels**: A list of corresponding class labels.
- **annotations**: A list of dictionaries containing bounding box information and other metadata.

#### Usage
```python
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
polygons, labels, annotations = mask_to_polygons_multi(mask)
