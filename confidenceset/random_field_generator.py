import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def ramp_2D(dim, mag, direction=0, fwhm=0, std=1, truncate=4):
  nsubj = dim[0]

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  else: #horizontal
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2]),dim[1]).reshape(dim[2],dim[1]).transpose()
  mu = np.tile(mu_temp, nsubj).reshape(dim)


  # noise
  noise = np.random.randn(*dim) * std
  sigma = fwhm / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma, truncate=truncate)  #smoothing

  data = mu + noise

  return(data, mu)

def circular_2D(dim, mag, r, fwhm_noise=0, fwhm_signal=3, std=1, truncate=4):
  nsubj = dim[0]
  sigma = fwhm / np.sqrt(8 * np.log(2))

  # signal
  center = np.array([dim[1]//2, dim[2]//2])
  x, y = np.meshgrid(np.linspace(-1,1,dim[1]), np.linspace(-1,1,dim[2]))
  cx, cy = 0, 0
  circle = (np.sqrt( (x-cx)**2 + (y-cy)**2  ) <= r)

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  circle_smth = gaussian_filter(circle, sigma = sigma_signal, truncate=truncate) * mag
  mu = np.tile(circle_smth, nsubj).reshape(dim)

  # noise
  noise = np.random.randn(*dim) * std
  sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma_noise, truncate=truncate)  #smoothing

  data = mu + noise

  return(data, mu)