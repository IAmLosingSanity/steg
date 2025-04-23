#!/usr/bin/env python3

import cv2
from modules import text_to_bits, lsb_embed, compute_maxD, compute_psnr, compute_snr, plot_error_vs_capacity, lsb_extract

if __name__ == '__main__':
    original = cv2.imread('1.png')
    if original is None:
        print("Ошибка: не удалось загрузить 1.png")
    else:
        text = "secret message"
        data = text_to_bits(text)
        
        stego = lsb_embed(original, data)
        cv2.imwrite('stego.png', stego)
        print("Данные встроены и стегоизображение сохранено как stego.png")
        
        maxD = compute_maxD(original, stego)
        snr = compute_snr(original, stego)
        psnr = compute_psnr(original, stego)
        
        print("μ_maxD:", maxD)
        print("μ_SNR:", snr, "dB")
        print("μ_PSNR:", psnr)

        extracted_bits = lsb_extract(stego, len(data))
        extracted_text = ''.join(chr(int(extracted_bits[i:i+8], 2)) for i in range(0, len(extracted_bits), 8))
        print("Извлечённый текст:", extracted_text)
        
        plot_error_vs_capacity(original, text)
