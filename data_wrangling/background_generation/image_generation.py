import os
import random
import argparse
from diffusers import DiffusionPipeline
import torch

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
parser.add_argument("--height", type=int, default=1024, help="Height of the generated images (default: 1024)")
parser.add_argument("--width", type=int, default=1280, help="Width of the generated images (default: 1280)")
parser.add_argument("--num_prompts", type=int, default=350, help="Number of images to generate per prompt (default: 350)")
parser.add_argument("--output_dir", type=str, default="generated_background_images", help="Directory to save generated images (default: 'generated_background_images')")
parser.add_argument("--prompts_file", type=str, default="prompts.txt", help="Path to the file containing prompts (default: 'prompts.txt')")
parser.add_argument("--seed", type=int, default=42, help="Random seed for image generation (default: 42)")

args = parser.parse_args()

# Load the diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("No GPU")

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Read prompts from the specified file
with open(args.prompts_file, "r") as file:
    prompts = file.readlines()

# Set image dimensions and number of prompts from arguments
height = args.height
width = args.width
n_prompt = args.num_prompts

# Set the random seed
random.seed(args.seed)

# Generate images for each prompt and save them
for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  # Remove any leading/trailing whitespace/newlines
    file_path = os.path.join(args.output_dir, f'prompt{i+1}')
    os.makedirs(file_path, exist_ok=True)
    
    if prompt:  # Ensure the prompt is not empty
        print(f"Generating images for prompt {i + 1}: {prompt}")
        
        for j in range(n_prompt):  # Generate specified number of images per prompt
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed for each image
            image_path = os.path.join(file_path, f"generated_image_{i + 1}_{j + 1}.png")
            if os.path.exists(image_path): 
                continue
            
            image = pipe(prompt=prompt, height=height, width=width,
                         num_inference_steps=50,
                         generator=torch.manual_seed(seed)).images[0]
            
            # Save the generated image 
            image.save(image_path)

print("Image generation complete.")