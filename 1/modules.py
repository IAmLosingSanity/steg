import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt

def text_to_bits(text):
    """Преобразует текст в битовую строку (8 бит на символ)."""
    return ''.join(format(ord(c), '08b') for c in text)

def lsb_embed(image, data):
    """
    Встраивает битовую строку data в изображение image, используя 1 младший бит каждого пикселя.
    
    Параметры:
    - image: исходное изображение (numpy-массив).
    - data: битовая строка, которую требуется встроить.
    
    Возвращает:
    - stego: изображение со встроенными данными.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    capacity = h * w * channels  # Используем только 1 бит на пиксель

    if len(data) > capacity:
        raise ValueError("Объём данных превышает ёмкость изображения.")

    flat = image.flatten()
    new_flat = flat.copy()
    
    for i in range(len(data)):
        new_flat[i] = (new_flat[i] & 0b11111110) | int(data[i])  # Заменяем только 1 младший бит
    
    stego = new_flat.reshape(image.shape)
    return stego

def lsb_extract(stego, data_length):
    """
    Извлекает битовую строку из стегоизображения, используя только 1 младший бит каждого пикселя.

    Параметры:
    - stego: изображение со встроенными данными.
    - data_length: общее количество бит, которое нужно извлечь.

    Возвращает:
    - Извлечённая битовая строка.
    """
    flat = stego.flatten()
    extracted_bits = ''.join(str(pixel & 1) for pixel in flat[:data_length])  # Извлекаем только 1 бит на пиксель

    return extracted_bits

def compute_snr(original, stego):
    """Вычисляет отношение сигнал/шум (SNR) в децибелах между оригиналом и стегоизображением."""
    signal = np.sum(original.astype(np.float64) ** 2)
    noise = np.sum((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if noise == 0:
        return float('inf')
    snr = 10 * np.log10(signal / noise)
    #snr = signal / noise
    return snr

def compute_maxD(original, stego):
    """Вычисляет максимальное абсолютное отклонение."""
    return np.max(np.abs(original.astype(np.float64) - stego.astype(np.float64)))

def compute_psnr(original, stego):
    """Вычисляет PSNR по формуле из методички."""
    max_pixel_value = float(np.max(original))
    mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return (float(original.shape[0] * original.shape[1]) * (max_pixel_value ** 2)) / mse


def plot_error_vs_capacity(original, text):
    """Строит график зависимости вероятности ошибок от объема скрываемых данных."""
    data_lengths = np.linspace(0, len(text_to_bits(text)), 10, dtype=int)
    errors = []
    
    for data_length in data_lengths:
        data = text_to_bits(text[:data_length // 8])
        stego = lsb_embed(original, data)
        extracted_bits = lsb_extract(stego, len(data))
        extracted_text = ''.join(chr(int(extracted_bits[i:i+8], 2)) for i in range(0, len(extracted_bits), 8))
        error_rate = sum(a != b for a, b in zip(text[:len(extracted_text)], extracted_text)) / len(text)
        errors.append(error_rate)
    
    plt.plot(data_lengths, errors, marker='o')
    plt.xlabel("Объем скрываемой информации (биты)")
    plt.ylabel("Вероятность ошибки")
    plt.title("Зависимость вероятности ошибок от объема скрываемых данных")
    plt.grid()
    plt.show()
