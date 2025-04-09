# steg_module.py
import numpy as np
import cv2
from math import ceil

def compute_maxD(original, stego):
    """
    Вычисляет максимальное абсолютное отклонение (μ_maxD)
    между оригинальным изображением и стего-изображением.
    """
    return np.max(np.abs(original.astype(np.float64) - stego.astype(np.float64)))

def compute_mse(original, stego):
    """
    Вычисляет среднюю квадратическую ошибку (MSE, μ_MSE)
    между оригинальным и стего-изображением.
    """
    mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    return mse

def compute_lp_norm(original, stego, p=4):
    """
    Вычисляет норму Минковского (μ_{L_p}) между оригинальным и стего, по формуле:
    
        μ_{L_p}(I, Ĩ) = ( (1/(W·H)) * Σ |I(x,y) - Ĩ(x,y)|^p )^(1/p)
        
    По умолчанию p=4, можно передавать другие значения p (например, 2).
    """
    diff = original.astype(np.float64) - stego.astype(np.float64)
    diff_p = np.abs(diff) ** p
    mean_diff_p = np.mean(diff_p)
    return mean_diff_p ** (1.0 / p)

def embed_data_brindox(container_img, secret_data, key=1234):
    """
    Встраивает секретные данные в изображение с использованием
    псевдослучайной перестановки пикселей (вариант алгоритма Брайндокса).
    
    Параметры:
        container_img : np.ndarray
            Изображение-контейнер (градации серого или один из каналов цветного изображения).
        secret_data : bytes
            Данные для встраивания (байтовая строка).
        key : int
            Ключ для псевдослучайной перестановки позиций.
    
    Возвращает:
        stego_img : np.ndarray
            Изображение со встроенными данными.
    """
    stego_img = container_img.copy()
    h, w = stego_img.shape[:2]
    
    # Преобразуем секретные данные в битовый массив
    bit_array = np.unpackbits(np.frombuffer(secret_data, dtype=np.uint8))
    
    # Генерация псевдослучайной перестановки индексов с использованием key
    np.random.seed(key)
    total_pixels = h * w
    indices = np.arange(total_pixels)
    np.random.shuffle(indices)
    
    # Проверяем, чтобы контейнер мог вместить все биты (1 бит на пиксель)
    if len(bit_array) > total_pixels:
        raise ValueError("Секретных бит больше, чем пикселей контейнера (один бит на пиксель).")
    
    flat_stego = stego_img.reshape(-1)
    for i, bit in enumerate(bit_array):
        pixel_index = indices[i]
        # Меняем самый младший бит пикселя на бит секретного сообщения
        flat_stego[pixel_index] = (flat_stego[pixel_index] & 0xFE) | bit
        
    stego_img = flat_stego.reshape(stego_img.shape)
    return stego_img

def extract_data_brindox(stego_img, data_length_bytes, key=1234):
    """
    Извлекает встроенные данные из стего-изображения, если известна длина сообщения.
    
    Параметры:
        stego_img : np.ndarray
            Изображение со встроенными данными.
        data_length_bytes : int
            Количество байтов, которое необходимо извлечь.
        key : int
            Ключ, использованный при встраивании.
    
    Возвращает:
        extracted_data : bytes
            Извлечённые данные (байтовая строка).
    """
    h, w = stego_img.shape[:2]
    total_pixels = h * w
    bits_needed = data_length_bytes * 8
    
    np.random.seed(key)
    indices = np.arange(total_pixels)
    np.random.shuffle(indices)
    
    flat_stego = stego_img.reshape(-1)
    extracted_bits = []
    for i in range(bits_needed):
        pixel_index = indices[i]
        bit = flat_stego[pixel_index] & 0x01
        extracted_bits.append(bit)
    
    bit_array = np.array(extracted_bits, dtype=np.uint8)
    byte_array = np.packbits(bit_array)
    extracted_data = byte_array.tobytes()
    return extracted_data

def test_jpeg_compression(stego_img, quality_list, data_length_bytes, key, original_secret_data):
    """
    Тестирует устойчивость стего-изображения к JPEG-сжатию.
    
    Для каждого значения качества сжимает изображение, затем извлекает
    встроенные данные и вычисляет BER (Bit Error Rate) – процент ошибочно извлечённых бит.
    
    Параметры:
        stego_img : np.ndarray
            Стего-изображение.
        quality_list : list[int]
            Список значений качества JPEG (например, [100, 90, 70, 50, 30, 10]).
        data_length_bytes : int
            Длина секретного сообщения в байтах.
        key : int
            Ключ для перестановки, использованный при встраивании.
        original_secret_data : bytes
            Оригинальное секретное сообщение для сравнения.
    
    Возвращает:
        results : dict
            Словарь, где для каждого значения качества записаны извлечённые данные и BER.
    """
    results = {}
    bit_array_original = np.unpackbits(np.frombuffer(original_secret_data, dtype=np.uint8))
    
    for q in quality_list:
        compressed_filename = f"temp_q{q}.jpg"
        cv2.imwrite(compressed_filename, stego_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        compressed_stego = cv2.imread(compressed_filename, cv2.IMREAD_GRAYSCALE)
        extracted = extract_data_brindox(compressed_stego, data_length_bytes, key)
        bit_array_extracted = np.unpackbits(np.frombuffer(extracted, dtype=np.uint8))
        
        # Если извлечённый массив оказался меньше, дополним его нулями
        if len(bit_array_extracted) < len(bit_array_original):
            bit_array_extracted = np.concatenate([bit_array_extracted, 
                                                  np.zeros(len(bit_array_original) - len(bit_array_extracted), dtype=np.uint8)])
        bit_errors = np.sum(bit_array_extracted != bit_array_original)
        ber_value = bit_errors / len(bit_array_original)
        results[q] = {"extracted_data": extracted, "BER": ber_value}
    return results
