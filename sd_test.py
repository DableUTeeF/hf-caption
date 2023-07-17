from diffusers import DiffusionPipeline
import torch
import warnings
warnings.filterwarnings('ignore')

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipeline.to("cuda")
# pipeline("An image of a squirrel in Picasso style").images[0]
