import cv2
import os
from modules import (
    embed_data_brindox, 
    extract_data_brindox, 
    compute_maxD, 
    compute_mse, 
    compute_lp_norm, 
    test_jpeg_compression
)

def main():
    # Укажите путь к изображению-контейнеру (например, "container.png")
    container_path = 'container.png'
    if not os.path.exists(container_path):
        print(f"Файл {container_path} не найден. Проверьте путь к изображению.")
        return
    
    # Читаем изображение в оттенках серого
    container_img = cv2.imread(container_path, cv2.IMREAD_GRAYSCALE)
    if container_img is None:
        print("Не удалось прочитать изображение. Проверьте формат файла.")
        return
    
    # Задаем секретное сообщение (его можно также прочитать из файла)
    secret_text = "Секретное сообщение для встраивания"
    secret_data = secret_text.encode('utf-8')
    data_length_bytes = len(secret_data)
    
    # Задаем ключ для псевдослучайной перестановки
    key = 2023

    # Встраиваем секретное сообщение в изображение
    stego_img = embed_data_brindox(container_img, secret_data, key)
    cv2.imwrite("stego.png", stego_img)
    print("Стего-изображение сохранено как 'stego.png'.")
    
    # Вычисляем метрики искажений между исходным и стего-изображением
    max_d = compute_maxD(container_img, stego_img)
    mse_val = compute_mse(container_img, stego_img)
    lp_norm  = compute_lp_norm(container_img, stego_img, p=4)
    
    print("\nМетрики искажений:")
    print("μ_maxD =", max_d)
    print("μ_MSE  =", mse_val)
    print("μ_Lp   =", lp_norm)
    
    # Проверяем корректность извлечения секретного сообщения
    extracted_data = extract_data_brindox(stego_img, data_length_bytes, key)
    try:
        extracted_text = extracted_data.decode('utf-8')
    except UnicodeDecodeError:
        extracted_text = "<Ошибка декодирования>"
    print("\nИзвлечённое сообщение:")
    print(extracted_text)
    
    # Тестируем устойчивость встроенной информации к JPEG-сжатию
    quality_list = [100, 90, 70, 50, 30, 10]
    compression_results = test_jpeg_compression(stego_img, quality_list, data_length_bytes, key, secret_data)
    
    print("\nРезультаты тестирования устойчивости при JPEG-сжатии:")
    for quality, result in compression_results.items():
        print(f"Качество {quality}: BER = {result['BER']*100:.2f}%")
    
if __name__ == "__main__":
    main()
