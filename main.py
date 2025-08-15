import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Загружаем модель в зависимости от доступного устройства
if device == "cuda":
    stableDiffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    stableDiffusion = stableDiffusion.to("cuda")
else:
    # Для CPU используем обычную версию без fp16
    stableDiffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    stableDiffusion = stableDiffusion.to("cpu")


def createImagesStableDiffusion(prompt='', rows=2, cols=2, iteration=20):
  # Запускаем генерацию
  images =  stableDiffusion([prompt] * (rows*cols), num_inference_steps=iteration).images
  w, h = images[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(images):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  
  # Сохраняем изображение вместо display
  output_path = "demultiplexer_chip.png"
  grid.save(output_path)
  print(f"Изображение сохранено: {output_path}")
  return grid

createImagesStableDiffusion("High-quality realistic photo of an electronic demultiplexer chip on a white background. The chip should have visible pins, labeled inputs (IN) and outputs (OUT), and a small manufacturer logo. Style: professional product photography with clean lighting and sharp focus. Details: matte surface, metallic pins, slight reflection. Mood: technical and precise.")