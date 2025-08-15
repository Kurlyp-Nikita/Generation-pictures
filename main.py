import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline():
	if device == "cuda":
		pipe = StableDiffusionPipeline.from_pretrained(
			"CompVis/stable-diffusion-v1-4",
			revision="fp16",
			torch_dtype=torch.float16,
		)
	else:
		# На CPU используем float32
		pipe = StableDiffusionPipeline.from_pretrained(
			"CompVis/stable-diffusion-v1-4",
		)
	return pipe.to(device)

stableDiffusion = load_pipeline()


def createImagesStableDiffusion(prompt: str = "", rows: int = 2, cols: int = 2, iteration: int = 20) -> Image.Image:
	# Генерируем изображения по одному, чтобы снизить потребление VRAM
	images = []
	num_images = rows * cols
	for _ in range(num_images):
		result = stableDiffusion(
			prompt,
			num_inference_steps=iteration,
		).images[0]
		images.append(result)

	w, h = images[0].size
	grid = Image.new("RGB", size=(cols * w, rows * h))

	for i, img in enumerate(images):
		grid.paste(img, box=((i % cols) * w, (i // cols) * h))
	return grid


if __name__ == "__main__":
	# Ваш промпт с пауком
	prompt = "Ultra-detailed macro photo of a female black widow spider (Latrodectus mactans) on a dark silk web. Glossy black body with vibrant red hourglass marking, dewdrops on the web, shallow depth of field. Photorealistic, studio lighting with soft shadows. Background: blurred forest at night. Mood: mesmerizing danger. Close-up of fangs with venom droplets. Golden backlight for dramatic effect."
	
	print("Генерирую изображения...")
	grid = createImagesStableDiffusion(prompt=prompt, rows=2, cols=2, iteration=20)
	output_path = "spider_grid.png"
	grid.save(output_path)
	print(f"Сохранено: {output_path}")