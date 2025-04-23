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

# ------------------ Алгоритм Брайндокса ------------------
BLOCK_SIZE = 8

def classify_groups(block: np.ndarray, slope_thresh: float = 5.0) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    flat = block.flatten()
    idx = np.argsort(flat)
    sorted_vals = flat[idx]
    diffs = np.diff(sorted_vals)
    max_i = np.argmax(diffs)
    if diffs[max_i] > slope_thresh:
        g1_idx = idx[:max_i+1]; g2_idx = idx[max_i+1:]
    else:
        half = len(flat) // 2
        g1_idx = idx[:half]; g2_idx = idx[half:]
    coords = [(i // BLOCK_SIZE, i % BLOCK_SIZE) for i in range(BLOCK_SIZE*BLOCK_SIZE)]
    return [coords[i] for i in g1_idx], [coords[i] for i in g2_idx]


def make_masks(key: int) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(key)
    mask = np.random.rand(BLOCK_SIZE, BLOCK_SIZE)
    return mask < 0.5, mask >= 0.5


def embed_bit(block: np.ndarray, bit: int, key: int, slope_thresh: float = 5.0) -> np.ndarray:
    g1, g2 = classify_groups(block, slope_thresh)
    maskA, maskB = make_masks(key)
    def avg(zone, cat): return np.array([block[x,y] for x,y in zone if cat[x,y]], dtype=np.float64)
    l11, l12 = avg(g1, maskA), avg(g1, maskB)
    l21, l22 = avg(g2, maskA), avg(g2, maskB)
    mu11, mu12 = (l11.mean() if l11.size else 0), (l12.mean() if l12.size else 0)
    mu21, mu22 = (l21.mean() if l21.size else 0), (l22.mean() if l22.size else 0)
    delta = (abs(mu11-mu12) + abs(mu21-mu22)) / 4 * (1 if bit else -1)
    stego = block.astype(np.float64).copy()
    for zone in (g1, g2):
        for x, y in zone: stego[x,y] += delta
    return np.clip(stego, 0, 255).astype(np.uint8)


def embed_message(img: np.ndarray, bits: List[int], key: int) -> np.ndarray:
    H, W = img.shape
    stego = img.copy()
    idx = 0
    for i in range(0, H, BLOCK_SIZE):
        for j in range(0, W, BLOCK_SIZE):
            if idx >= len(bits): break
            block = stego[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
                stego[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = embed_bit(block, bits[idx], key + idx)
                idx += 1
        if idx >= len(bits): break
    return stego


def extract_message(stego: np.ndarray, length: int, key: int, slope_thresh: float = 5.0) -> List[int]:
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
            d1 = np.mean([block[x,y] for x,y in g1 if maskA[x,y]] or [0]) - np.mean([block[x,y] for x,y in g1 if maskB[x,y]] or [0])
            d2 = np.mean([block[x,y] for x,y in g2 if maskA[x,y]] or [0]) - np.mean([block[x,y] for x,y in g2 if maskB[x,y]] or [0])
            bits.append(1 if (d1 + d2) > 0 else 0)
            idx += 1
        if idx >= length: break
    return bits

# ------------------ Тестирование и сохранение результатов ------------------
def test_stego(image_path: str, message: bytes, key: int = 42, out_dir: str = 'results'):
    os.makedirs(out_dir, exist_ok=True)
    orig_bgr = cv2.imread(image_path)
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, 'orig_gray.png'), orig_gray)

    max_blocks = (orig_gray.shape[0]//BLOCK_SIZE)*(orig_gray.shape[1]//BLOCK_SIZE)
    bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))[:max_blocks]

    stego = embed_message(orig_gray, bits.tolist(), key)
    cv2.imwrite(os.path.join(out_dir, 'stego.png'), stego)

    mu_maxD = compute_maxD(orig_gray, stego)
    mu_mse = compute_mse(orig_gray, stego)
    mu_lp = compute_lp_norm(orig_gray, stego, p=4)
    print(f"μ_maxD={mu_maxD:.4f}, μ_MSE={mu_mse:.4f}, μ_L4={mu_lp:.4f}")

    qualities = [90, 70, 50, 30]
    for q in qualities:
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        fname_jpg = os.path.join(out_dir, f'stego_q{q}.jpg')
        cv2.imwrite(fname_jpg, stego, enc_param)
        comp = cv2.imread(fname_jpg, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(out_dir, f'comp_gray_q{q}.png'), comp)
        extracted = extract_message(comp, len(bits), key)
        ber = np.mean(np.abs(bits - np.array(extracted)))
        print(f"Quality={q}: BER={ber:.4f}")

if __name__ == '__main__':
    test_stego('photomode_12042025_211937.png', b'Hidden message for stego', key=1234)
