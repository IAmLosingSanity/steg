import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt

def text_to_bits(text):
    """Преобразует текст в битовую строку (8 бит на символ)."""
    return ''.join(format(ord(c), '08b') for c in text)

def lsb_embed(image, data):
    """
    Встраивает битовую строку data в изображение image, используя два младших бита каждого пикселя.
    
    Параметры:
    - image: исходное изображение (numpy-массив).
    - data: битовая строка, которую требуется встроить.
    
    Возвращает:
    - stego: изображение со встроенными данными.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    capacity = h * w * channels * 2
    
    if len(data) > capacity:
        raise ValueError("Объём данных превышает ёмкость изображения.")

    flat = image.flatten()
    new_flat = flat.copy()
    data_index = 0

    for i in range(len(flat)):
        if data_index < len(data):
            bits_to_embed = data[data_index:data_index+2]
            if len(bits_to_embed) < 2:
                bits_to_embed = bits_to_embed + '0'
            data_index += 2

            new_flat[i] = (new_flat[i] & 0b11111100) | int(bits_to_embed, 2)

        else:
            break

    stego = new_flat.reshape(image.shape)
    return stego

def lsb_extract(stego, data_length):
    """
    Извлекает битовую строку из стегоизображения.
    
    Параметры:
    - stego: изображение со встроенными данными.
    - data_length: общее количество бит, которое нужно извлечь.
    
    Возвращает:
    - Извлечённая битовая строка.
    """
    flat = stego.flatten()
    extracted_bits = ''
    for pixel in flat:
        if len(extracted_bits) >= data_length:
            break
        extracted_bits += format(pixel & 3, '02b')
    return extracted_bits[:data_length]

def compute_mse(original, stego):
    """Вычисляет среднюю квадратическую ошибку (MSE) между оригинальным и стегоизображением."""
    mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    return mse

def compute_snr(original, stego):
    """Вычисляет отношение сигнал/шум (SNR) в децибелах между оригиналом и стегоизображением."""
    signal = np.sum(original.astype(np.float64) ** 2)
    noise = np.sum((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if noise == 0:
        return float('inf')
    snr = 10 * np.log10(signal / noise)
    #snr = signal / noise
    return snr

def compute_lmse(original, stego):
    """
    Вычисляет LMSE по формуле:
    μ_LMSE(I, Ĩ) = (Σ(∆²I - ∆²Ĩ)²) / (Σ(∆²I)²).
    Если исходное изображение однородно (лапласиан равен 0), вернётся 0 или 'inf' (на выбор),
    поскольку знаменатель может оказаться равным 0.
    """
    if original.ndim == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        stego_gray = stego

    laplacian_original = cv2.Laplacian(original_gray.astype(np.float64), cv2.CV_64F)
    laplacian_stego = cv2.Laplacian(stego_gray.astype(np.float64), cv2.CV_64F)

    numerator = np.sum((laplacian_original - laplacian_stego) ** 2)
    denominator = np.sum(laplacian_original ** 2)

    if denominator == 0:
        # Если лапласиан оригинала везде 0 (абсолютно однородное изображение),
        # можно вернуть 0, так как формально нет "текстуры", которую можно исказить.
        return 0

    lmse = numerator / denominator
    return lmse

def compute_maxD(original, stego):
    """Вычисляет максимальное абсолютное отклонение."""
    return np.max(np.abs(original.astype(np.float64) - stego.astype(np.float64)))

def compute_psnr(original, stego):
    """Вычисляет PSNR по формуле из методички."""
    max_pixel_value = np.max(original)
    mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return original.shape[0] * original.shape[1] * (max_pixel_value ** 2) / mse

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



if __name__ == '__main__':
    original = cv2.imread('1.png')
    if original is None:
        print("Ошибка: не удалось загрузить .png")
    else:
        text = "a little secret is right here"
        data = text_to_bits(text)
        # encoded_text = base64.b64encode(text.encode('utf-8')).decode('ascii')  # Base64-кодирование
        # print(encoded_text)

        stego = lsb_embed(original, data)
        cv2.imwrite('stego.png', stego)
        print("Данные встроены и стегоизображение сохранено как stego.png")

        mse = compute_mse(original, stego)
        snr = compute_snr(original, stego)
        lmse = compute_lmse(original, stego)

        print("μ_MSE:", mse)
        print("μ_SNR:", snr, "dB")
        print("μ_LMSE:", lmse)

        extracted_bits = lsb_extract(stego, len(data))
        extracted_text = ''.join(chr(int(extracted_bits[i:i+8], 2)) for i in range(0, len(extracted_bits), 8))
        print("Извлечённый текст:", extracted_text)
        # decoded_text = base64.b64decode(extracted_text.encode('ascii')).decode('utf-8')
        # print(decoded_text)
