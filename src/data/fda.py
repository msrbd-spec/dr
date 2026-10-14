# File: src/data/fda.py
import numpy as np
import cv2

def apply_fft_amplitude_swap(src_img: np.ndarray, tgt_img: np.ndarray, beta: float = 0.03) -> np.ndarray:
    """
    Applies Fourier Domain Adaptation (FDA) by swapping the low-frequency 
    amplitude spectrum of the source image with the target image.
    Must be applied to RAW images before CLAHE/Ben Graham.
    """
    h, w, c = src_img.shape
    tgt_img_resized = cv2.resize(tgt_img, (w, h), interpolation=cv2.INTER_AREA)

    # Convert to float32 and transpose to (C, H, W) for FFT
    src = src_img.transpose(2, 0, 1).astype(np.float32)
    tgt = tgt_img_resized.transpose(2, 0, 1).astype(np.float32)

    # Compute 2D FFT
    fft_src = np.fft.fft2(src, axes=(-2, -1))
    fft_tgt = np.fft.fft2(tgt, axes=(-2, -1))

    # Extract amplitude and phase
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_tgt = np.abs(fft_tgt)

    # Shift zero-frequency component to the center
    amp_src_shift = np.fft.fftshift(amp_src, axes=(-2, -1))
    amp_tgt_shift = np.fft.fftshift(amp_tgt, axes=(-2, -1))

    # Define the center bounding box based on beta
    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2

    # Swap the low-frequency amplitude
    amp_src_shift[:, cy - b:cy + b, cx - b:cx + b] = amp_tgt_shift[:, cy - b:cy + b, cx - b:cx + b]

    # Inverse shift and reconstruct
    amp_mixed = np.fft.ifftshift(amp_src_shift, axes=(-2, -1))
    fft_mixed = amp_mixed * np.exp(1j * pha_src)
    
    # Inverse FFT to get spatial domain image
    mixed_img = np.real(np.fft.ifft2(fft_mixed, axes=(-2, -1)))
    
    # Clip and convert back to uint8
    mixed_img = np.clip(mixed_img, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    
    return mixed_img