import cv2
import numpy as np

# Загружаем оригинал и стего-изображение
orig = cv2.imread("1.png")
stego = cv2.imread("2.png")

# Разница в младших 2 битах (по каждому каналу)
lsb_diff = ((orig ^ stego) & 0b00000011) * 85  # Умножаем на 85, чтобы сделать разницу видимой

# Показываем результат
cv2.imshow("LSB Difference", lsb_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
