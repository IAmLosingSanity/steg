from PIL import Image, ImageChops

# Загрузка изображений
image1_path = "1.png"
image2_path = "stego.png"

image1 = Image.open(image1_path).convert("RGB")
image2 = Image.open(image2_path).convert("RGB")

# Поиск разницы между изображениями
diff = ImageChops.difference(image1, image2)

# Проверяем, есть ли вообще различия
bbox = diff.getbbox()

# Сохраняем результат, если есть различия
if bbox:
    diff_path = "diff.png"
    diff.save(diff_path)
    result = diff_path
else:
    result = "Изображения идентичны."

result
