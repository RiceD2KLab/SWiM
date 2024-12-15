import argparse
import yaml
import os

def main(args):
    # Load the YAML content
    yaml_path = args.yaml_path
    dataset_path = args.dataset_path
    augment = args.augment

    yaml_content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': {
            0: 'seep',
        },
        'patience': 20
    }


    if augment:
        yaml_content.update({
            'hsv_h': 0.015,  # Hue augmentation
            'hsv_s': 0.7,    # Saturation augmentation
            'hsv_v': 0.4,    # Value (brightness) augmentation
            'degrees': 90,    # Image rotation (+/- deg)
            'translate': 0.1,  # Image translation (+/- fraction)
            'scale': 0.5,    # Image scale (+/- gain)
            'shear': 0.0,    # Image shear (+/- deg)
            'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
            'flipud': 0.5,   # Image flip up-down (probability)
            'fliplr': 0.5,   # Image flip left-right (probability)
            'mosaic': 1.0,   # Image mosaic (probability)
            'mixup': 0.5,    # Image mixup (probability)
            'copy_paste': 0.5,  # Segment copy-paste (probability)
        })

    # Save the YAML content
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML file for YOLO training')
    parser.add_argument("--yaml-path", required=True, help="Path to save the YAML file")
    parser.add_argument("--dataset-path", required=True, help="Path to the dataset directory")
    parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
    args = parser.parse_args()
    main(args)