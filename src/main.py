#!/usr/bin/env python3

import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft

# Select the first GPU
cp.cuda.runtime.setDevice(0)

"""Perform Fast Fourier Transform on GPU"""

def gpu_fft(arr):
    with scipy.fft.set_backend(cufft):
        fft_arr = scipy.fft.fft(arr)
    return fft_arr

n_bytes = 192

arr = cp.random.random(n_bytes).astype(cp.float32)

fft_arr = gpu_fft(arr)

print(fft_arr)
