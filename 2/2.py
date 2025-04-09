import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Функции оценки искажений
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

# Функции встраивания и извлечения для изображения
def embed_watermark(image_array, watermark, lam):
    """
    Встраивает watermark в изображение с использованием параметра lam.
    Здесь используется аддитивная модель: стего = изображение + lam * watermark.
    Если изображение цветное, применяется ко всем каналам.
    """
    # Приводим к типу float для точности
    stego = image_array.astype(np.float64) + lam * watermark.astype(np.float64)
    # Обрезаем значения до допустимого диапазона [0, 255]
    stego = np.clip(stego, 0, 255)
    return stego

def extract_watermark(original, stego, lam):
    """
    Извлекает watermark из стегоизображения с использованием оригинального изображения и lam.
    Вычисляется как (stego - original)/lam с последующей бинаризацией по порогу 0.
    """
    extracted = (stego - original.astype(np.float64)) / lam
    extracted = np.where(extracted >= 0, 1, -1)
    return extracted

def simulate_with_image(image_path, lam_values, num_trials=10):
    """
    Для заданного набора значений lam проводит серию испытаний с использованием изображения,
    вычисляя вероятность ошибки извлечения и метрики искажений.
    Возвращает результаты в виде словаря.
    """
    # Загружаем изображение и приводим к numpy-массиву
    img = Image.open(image_path)
    image_array = np.array(img)
    
    # Если изображение цветное, обработаем каждый канал отдельно
    if image_array.ndim == 3:
        channels = image_array.shape[2]
    else:
        channels = 1
        image_array = image_array[..., np.newaxis]  # добавляем размерность для единообразия

    results = {lam: [] for lam in lam_values}
    
    # Генерируем watermark для каждого канала (матрица тех же размеров, что и изображение)
    watermark = np.random.choice([1, -1], size=image_array.shape)
    
    for lam in lam_values:
        error_rates = []
        mses = []
        snrs = []
        maxDs = []
        
        for _ in range(num_trials):
            # Встраивание watermark в изображение
            stego = embed_watermark(image_array, watermark, lam)
            
            # Извлечение watermark
            extracted = extract_watermark(image_array, stego, lam)
            
            # Вычисление вероятности ошибки (сравниваем по всем каналам)
            errors = np.sum(extracted != watermark)
            total = watermark.size
            err_rate = errors / total
            
            # Метрики искажений (считаем по объединённому изображению)
            mse = compute_mse(image_array, stego)
            snr = compute_snr(image_array, stego)
            maxD = compute_maxD(image_array, stego)
            
            error_rates.append(err_rate)
            mses.append(mse)
            snrs.append(snr)
            maxDs.append(maxD)
        
        results[lam] = {
            'err_rate': np.mean(error_rates),
            'mse': np.mean(mses),
            'snr': np.mean(snrs),
            'maxD': np.mean(maxDs)
        }
        
    return results, image_array, watermark

# Путь к изображению (замените на актуальный путь)
image_path = '1.png'

# Задаём диапазон lam
lam_values = np.linspace(0.1, 5, 20)
results, original_image, watermark = simulate_with_image(image_path, lam_values, num_trials=10)

# Извлекаем данные для построения графика зависимости вероятности ошибки от lam
err_rates = [results[lam]['err_rate'] for lam in lam_values]

plt.figure(figsize=(8, 6))
plt.plot(lam_values, err_rates, marker='o', linestyle='-')
plt.xlabel('λ (энергия встраиваемого сигнала)')
plt.ylabel('Вероятность ошибки извлечения')
plt.title('Зависимость вероятности ошибок извлечения от λ')
plt.grid(True)
plt.show()

# Выводим метрики для каждого значения lam
for lam in lam_values:
    print(f"λ = {lam:.2f} | Err rate = {results[lam]['err_rate']:.4f} | MSE = {results[lam]['mse']:.2f} | SNR = {results[lam]['snr']:.2f} dB | maxD = {results[lam]['maxD']:.2f}")

# В качестве примера можно также визуализировать оригинальное и стего-изображение для выбранного lam
selected_lam = lam_values[len(lam_values) // 2]
stego_image = embed_watermark(original_image, watermark, selected_lam)

# Если изображение было цветным, преобразуем обратно, иначе убираем лишнюю размерность
if stego_image.shape[2] == 1:
    stego_image = stego_image.squeeze(axis=2)

# Отображаем оригинал и стего-изображение
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image.astype(np.uint8))
plt.title("Оригинальное изображение")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stego_image.astype(np.uint8))
plt.title(f"Stego изображение (λ = {selected_lam:.2f})")
plt.axis('off')

plt.show()

output_dir = "stego_images"
os.makedirs(output_dir, exist_ok=True)

# Выбираем λ для сохранения примера
# selected_lam = lam_values[len(lam_values) // 2]
selected_lam = lam_values[0]
stego_image = embed_watermark(original_image, watermark, selected_lam)

# Преобразуем к uint8 перед сохранением
stego_pil = Image.fromarray(stego_image.astype(np.uint8))

# Формируем путь к файлу
stego_path = os.path.join(output_dir, f"stego_lambda_{selected_lam:.2f}.png")

# Сохраняем изображение
stego_pil.save(stego_path)
print(f"Stego изображение сохранено в {stego_path}")
