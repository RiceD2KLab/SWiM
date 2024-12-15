import numpy as np


def yolo_polygon_to_xyxy(file_path, img_width=1280, img_height=1024):
    # Read the polygon coordinates from the file
    bboxes = []
    with open(file_path, 'r') as f:
        line = f.readline().strip().split()
    
    # Convert string coordinates to float and reshape into (n, 2) array
    polygon = np.array([float(x) for x in line[1:]]).reshape(-1, 2)
    
    # Convert normalized coordinates to pixel coordinates
    polygon[:, 0] *= img_width
    polygon[:, 1] *= img_height
    
    # Find min and max x, y coordinates
    xmin = np.min(polygon[:, 0])
    ymin = np.min(polygon[:, 1])
    xmax = np.max(polygon[:, 0])
    ymax = np.max(polygon[:, 1])
    
    bboxes.append([xmin, ymin, xmax, ymax])

    return np.array(bboxes)