import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Оценочные функции искажений

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
    return snr


def compute_maxD(original, stego):
    """Вычисляет максимальное абсолютное отклонение."""
    return np.max(np.abs(original.astype(np.float64) - stego.astype(np.float64)))

# Функции стеганографии Куттера-Джордана-Боссена

def embed_kjb(container, message_bits, lam):
    """
    Встраивает биты message_bits в синий канал контейнера по методу Куттера-Джордана-Боссена.
    container: HxWx3 uint8
    message_bits: 1D array из {0,1}
    lam: константа lambda (энергия встраиваемого сигнала)
    """
    stego = container.copy().astype(np.float64)
    H, W, _ = container.shape
    # Случайный порядок встраивания
    coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1).reshape(-1,2)
    np.random.seed(42)
    chosen = coords[np.random.choice(len(coords), size=len(message_bits), replace=False)]

    for (y, x), m in zip(chosen, message_bits):
        R, G, B = container[y, x]
        L = 0.299*R + 0.587*G + 0.114*B
        if m == 1:
            stego[y, x, 2] = B + lam * L
        else:
            stego[y, x, 2] = B - lam * L
    # Ограничение по диапазону
    stego = np.clip(stego, 0, 255).astype(np.uint8)
    return stego, chosen


def extract_kjb(stego, chosen_coords, lam, sigma=1):
    """
    Извлекает биты из синих каналов стегоизображения по методу Куттера.
    stego: HxWx3 uint8
    chosen_coords: список координат, в которых была встраивалась информация
    sigma: число соседних пикселей (обычно 1)
    """
    H, W, _ = stego.shape
    extracted = []
    for (y, x) in chosen_coords:
        # Предсказываемое значение синей компоненты
        neigh = []
        for dy, dx in [(-sigma,0),(sigma,0),(0,-sigma),(0,sigma)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                neigh.append(int(stego[ny, nx, 2]))
        if not neigh:
            neigh_val = stego[y, x, 2]
        else:
            neigh_val = sum(neigh) / len(neigh)
        # Сравнение
        if stego[y, x, 2] > neigh_val:
            extracted.append(1)
        else:
            extracted.append(0)
    return np.array(extracted, dtype=np.uint8)

# Демонстрация и оценка

def evaluate_on_image(image_path, lambdas, n_bits=5000, output_dir='stego_outputs'):
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    container = cv2.imread(image_path)
    if container is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    # Преобразование BGR -> RGB
    container = cv2.cvtColor(container, cv2.COLOR_BGR2RGB)
    H, W, _ = container.shape

    # Генерация случайного сообщения
    np.random.seed(0)
    message = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)

    # Массивы результатов
    results = {
        'lambda': [],
        'error_prob': [],
        'mse': [],
        'snr': [],
        'maxD': []
    }

    for lam in lambdas:
        stego, coords = embed_kjb(container, message, lam)

        # Сохраняем стегоизображение
        lam_str = str(lam).replace('.', '_')
        stego_bgr = cv2.cvtColor(stego, cv2.COLOR_RGB2BGR)
        stego_path = os.path.join(output_dir, f'stego_{lam_str}.png')
        cv2.imwrite(stego_path, stego_bgr)

        extracted = extract_kjb(stego, coords, lam)
        error_prob = np.mean(message != extracted)
        # Оценка искажений
        mse = compute_mse(container, stego)
        snr = compute_snr(container, stego)
        maxd = compute_maxD(container, stego)

        results['lambda'].append(lam)
        results['error_prob'].append(error_prob)
        results['mse'].append(mse)
        results['snr'].append(snr)
        results['maxD'].append(maxd)

        print(f"λ={lam}: Ошибки={error_prob:.4f}, MSE={mse:.2f}, SNR={snr:.2f} дБ, maxD={maxd}")

    # Построение зависимости P_error от λ
    plt.figure()
    plt.plot(results['lambda'], results['error_prob'], marker='o')
    plt.xlabel('λ')
    plt.ylabel('Вероятность ошибки')
    plt.title('Зависимость вероятности ошибки от λ')
    plt.grid(True)
    plt.show()

    return results

if __name__ == '__main__':
    # Пример использования
    lambdas = [0.1, 0.5, 1, 2, 5, 10]
    results = evaluate_on_image('container.png', lambdas)
