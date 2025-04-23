import numpy as np
import cv2
import os
from typing import List, Tuple

# ------------------ Метрики и вспомогательные функции ------------------
def compute_maxD(original: np.ndarray, stego: np.ndarray) -> float:
    return np.max(np.abs(original.astype(np.float64) - stego.astype(np.float64)))

def compute_mse(original: np.ndarray, stego: np.ndarray) -> float:
    return np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)

def compute_lp_norm(original: np.ndarray, stego: np.ndarray, p: int = 4) -> float:
    diff = np.abs(original.astype(np.float64) - stego.astype(np.float64)) ** p
    return np.mean(diff) ** (1.0 / p)

# ------------------ Алгоритм Брайндокса по методичке ------------------
BLOCK_SIZE = 8

def classify_groups(block: np.ndarray, slope_thresh: float = 5.0) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Классификация пикселей внутри блока на две группы по яркости.
    """
    flat = block.flatten()
    idx = np.argsort(flat)
    diffs = np.diff(flat[idx])
    max_i = np.argmax(diffs)
    if diffs[max_i] > slope_thresh:
        g1_idx = idx[:max_i+1]; g2_idx = idx[max_i+1:]
    else:
        half = len(flat) // 2
        g1_idx = idx[:half]; g2_idx = idx[half:]
    coords = [(i // BLOCK_SIZE, i % BLOCK_SIZE) for i in range(BLOCK_SIZE * BLOCK_SIZE)]
    return [coords[i] for i in g1_idx], [coords[i] for i in g2_idx]


def make_masks(key: int) -> Tuple[np.ndarray, np.ndarray]:
    """Генерация псевдослучайных масок A и B одинакового размера блока."""
    np.random.seed(key)
    rand = np.random.rand(BLOCK_SIZE, BLOCK_SIZE)
    return rand < 0.5, rand >= 0.5


def embed_bit_exact(block: np.ndarray, bit: int, key: int, delta: float = 1.0, slope_thresh: float = 5.0) -> np.ndarray:
    """
    Точное встраивание одного бита по методичке:
    - делим на группы g1,g2 и категории A,B
    - вычисляем исходные средние μ11,μ12,μ21,μ22
    - считаем объединённые μ̃1,μ̃2
    - задаём новые μ' сдвигом Δ
    - корректируем яркости в каждой категории
    """
    g1, g2 = classify_groups(block, slope_thresh)
    maskA, maskB = make_masks(key)

    def mean_vals(zone, mask):
        vals = [block[x, y] for x, y in zone if mask[x, y]]
        return np.mean(vals) if vals else 0.0

    μ11 = mean_vals(g1, maskA)
    μ12 = mean_vals(g1, maskB)
    μ21 = mean_vals(g2, maskA)
    μ22 = mean_vals(g2, maskB)

    n11 = sum(1 for x,y in g1 if maskA[x,y])
    n12 = sum(1 for x,y in g1 if maskB[x,y])
    n21 = sum(1 for x,y in g2 if maskA[x,y])
    n22 = sum(1 for x,y in g2 if maskB[x,y])

    μ_tilde1 = (n11*μ11 + n12*μ12) / (n11 + n12) if (n11 + n12) else 0
    μ_tilde2 = (n21*μ21 + n22*μ22) / (n21 + n22) if (n21 + n22) else 0

    # Определяем новые средние согласно биту
    sign = 1 if bit else -1
    μ11_p = μ_tilde1 + sign * delta
    μ12_p = μ_tilde1 - sign * delta
    μ21_p = μ_tilde2 + sign * delta
    μ22_p = μ_tilde2 - sign * delta

    # Расчёт, на сколько менять яркость каждой категории
    δ11 = μ11_p - μ11
    δ12 = μ12_p - μ12
    δ21 = μ21_p - μ21
    δ22 = μ22_p - μ22

    stego = block.astype(np.float64).copy()
    # Коррекция пикселей
    for (x, y) in g1:
        if maskA[x,y]: stego[x,y] += δ11
        else:          stego[x,y] += δ12
    for (x, y) in g2:
        if maskA[x,y]: stego[x,y] += δ21
        else:          stego[x,y] += δ22

    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_message(img: np.ndarray, bits: List[int], key: int = 0, delta: float = 1.0) -> np.ndarray:
    """
    Встраивает последовательность битов в изображение блочно (8×8).
    delta — уровень встраивания Δ.
    """
    H, W = img.shape
    stego = img.copy().astype(np.uint8)
    idx = 0
    for i in range(0, H, BLOCK_SIZE):
        for j in range(0, W, BLOCK_SIZE):
            if idx >= len(bits): break
            block = stego[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
                stego[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = embed_bit_exact(block, bits[idx], key + idx, delta)
                idx += 1
        if idx >= len(bits): break
    return stego


def extract_message(stego: np.ndarray, length: int, key: int = 0, slope_thresh: float = 5.0) -> List[int]:
    """Извлекает последовательность битов из стего-изображения."""
    H, W = stego.shape
    bits = []
    idx = 0
    for i in range(0, H, BLOCK_SIZE):
        for j in range(0, W, BLOCK_SIZE):
            if idx >= length: break
            block = stego[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            if block.shape != (BLOCK_SIZE, BLOCK_SIZE): continue
            g1, g2 = classify_groups(block, slope_thresh)
            maskA, maskB = make_masks(key + idx)
            def mean_zone(zone, mask):
                vals = [block[x,y] for x,y in zone if mask[x,y]]
                return np.mean(vals) if vals else 0.0
            d1 = mean_zone(g1, maskA) - mean_zone(g1, maskB)
            d2 = mean_zone(g2, maskA) - mean_zone(g2, maskB)
            bits.append(1 if (d1 + d2) > 0 else 0)
            idx += 1
        if idx >= length: break
    return bits

# ------------------ Тестирование и сохранение ------------------
def test_stego_exact(image_path: str, message: bytes, key: int = 0, delta: float = 1.0, out_dir: str = 'results_exact'):
    os.makedirs(out_dir, exist_ok=True)
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(out_dir, 'orig_gray.png'), orig)

    max_blocks = (orig.shape[0]//BLOCK_SIZE) * (orig.shape[1]//BLOCK_SIZE)
    bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))[:max_blocks]

    stego = embed_message(orig, bits.tolist(), key, delta)
    cv2.imwrite(os.path.join(out_dir, 'stego.png'), stego)

    mu_maxD = compute_maxD(orig, stego)
    mu_mse = compute_mse(orig, stego)
    mu_lp = compute_lp_norm(orig, stego, p=4)
    print(f"μ_maxD={mu_maxD:.4f}, μ_MSE={mu_mse:.4f}, μ_L4={mu_lp:.4f}")

    qualities = [90, 70, 50, 30]
    for q in qualities:
        fname = os.path.join(out_dir, f'stego_q{q}.jpg')
        cv2.imwrite(fname, stego, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        comp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(out_dir, f'comp_gray_q{q}.png'), comp)
        extracted = extract_message(comp, len(bits), key)
        ber = np.mean(np.abs(bits - np.array(extracted)))
        print(f"Quality={q}: BER={ber:.4f}")

if __name__ == '__main__':
    test_stego_exact('photomode_12042025_211937.png', b'Hidden message for stego', key=1234, delta=1.0)
