# Background Generation with Stable Diffusion

## Overview
The `image_generation.py` script generates images using the Stable Diffusion model. It allows to specify various parameters such as image dimensions, the number of images to generate per prompt, and the output directory for saving generated images. We can also provide a custom file containing prompts and set a random seed for reproducibility.

## Requirements
To run this script, make sure you have all the libraries installed enumerated in the requirements.txt file in the root directory. The critical libraries for this script is:


- `diffusers==0.31.0`

## Usage
To run the script, use the following command in your terminal:
```bash
python3 image_generation.py [OPTIONS]
```

### Arguments

The following arguments can be provided when running the script:
--height (int): Height of the generated images (default: 1024).
--width (int): Width of the generated images (default: 1280).
--num_prompts (int): Number of images to generate per prompt (default: 350).
--output_dir (str): Directory to save generated images (default: 'generated_background_images').
--prompts_file (str): Path to the file containing prompts (default: 'prompts.txt').
--seed (int): Random seed for image generation (default: 42).

### Example Command

```bash
python3 image_generation.py --height 1024 --width 1280 --num_prompts 100 --output_dir "my_generated_images" --prompts_file "my_prompts.txt" --seed 12345
```

## Notes

Ensure you have a compatible GPU and CUDA installed if you want to leverage GPU acceleration.
The prompts file should contain one prompt per line.
The output directory will be created if it does not already exist.

### License

This project is licensed under the Apache License - see the LICENSE file in root directory for more details.
