import os
import random
from diffusers import DiffusionPipeline
import torch


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("No GPU")
    
output_dir = "generated_background_images"
os.makedirs(output_dir, exist_ok=True)

# Read prompts from the text file
with open("prompts.txt", "r") as file:
    prompts = file.readlines()

height = 1024
width = 1280

random.seed(42)

n_prompt = 350

# Generate images for each prompt and save them
for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  # Remove any leading/trailing whitespace/newlines
    file_path = output_dir + f'/prompt{i+1}'
    os.makedirs(file_path, exist_ok=True)
    if prompt:  # Ensure the prompt is not empty
        print(f"Generating images for prompt {i + 1}: {prompt}")
        
        for j in range(n_prompt):  # Generate 10 images per prompt
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
#            print(f"Using seed: {seed}")
            image_path = os.path.join(file_path, f"generated_image_{i + 1}_{j + 1}.png")
            if os.path.exists(image_path): continue
            image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=50, generator=torch.manual_seed(seed)).images[0]
            
            # Save the generated image
 
            image.save(image_path)
#            print(f"Image saved to {image_path}")

print("Image generation complete.")