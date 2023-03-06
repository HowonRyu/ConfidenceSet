import numpy as np
from scipy.ndimage import gaussian_filter


def ramp_2D(dim, mag, direction=0, fwhm=0, std=1, truncate=4):
  nsubj = dim[0]

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  else: #horizontal
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2]),dim[1]).reshape(dim[2],dim[1]).transpose()
  mu = np.array(mu_temp, dtype='float')
  #mu = np.tile(mu_temp, nsubj).reshape(dim)


  # noise
  noise = np.random.randn(*dim) * std
  sigma = fwhm / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma, truncate=truncate)  #smoothing

  data = np.array(mu + noise, dtype='float')

  return(data, mu)


def circular_2D(dim, shape_spec, truncate=4):
  nsubj = dim[0]
  r = shape_spec['r']
  mag = shape_spec['mag']
  fwhm_noise = shape_spec['fwhm_noise']
  fwhm_signal = shape_spec['fwhm_signal']
  fwhm_signal = shape_spec['fwhm_signal']
  std = shape_spec['std']

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))

  # signal
  x, y = np.meshgrid(np.linspace(-1,1,dim[1]), np.linspace(-1,1,dim[2]))
  cx, cy = 0, 0
  circle = np.array((np.sqrt((x-cx)**2 + (y-cy)**2) <= r), dtype='float')

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  circle_smth = gaussian_filter(circle, sigma = sigma_signal, truncate=truncate)
  mu = np.array(circle_smth * mag, dtype='float')

  # noise
  noise = np.random.randn(*dim) * std
  sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma_noise, truncate=truncate)  #smoothing

  data = np.array(mu + noise, dtype='float')
  return(data, mu)



