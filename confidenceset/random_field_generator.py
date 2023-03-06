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



def conf_plot_agg(c, method, std = 5, tail="two", _min=0, _max=3, fontsize = 25, figsize=(30, 20)):
  dim_100 = (80,100,100)
  spec_cir_l100 = {'r':0.8, 'std':std,'mag':3, 'fwhm_noise':0, 'fwhm_signal':10*2 }
  spec_cir_s100 = {'r':0.45, 'std':std,'mag':3, 'fwhm_noise':0, 'fwhm_signal':10*2 }
  spec_cir_l100_smth = {'r':0.8, 'std':std,'mag':3, 'fwhm_noise':3*2, 'fwhm_signal':10*2 }
  spec_cir_s100_smth = {'r':0.45, 'std':std,'mag':3, 'fwhm_noise':3*2, 'fwhm_signal':10*2 }

  circular_l100 = circular_2D(dim=dim_100, shape_spec=spec_cir_l100)[0]
  circular_s100 = circular_2D(dim=dim_100, shape_spec=spec_cir_s100)[0]
  circular_l100_smth = circular_2D(dim=dim_100, shape_spec=spec_cir_l100_smth)[0]
  circular_s100_smth = circular_2D(dim=dim_100, shape_spec=spec_cir_s100_smth)[0]
  ramp_100 = ramp_2D(dim=dim_100, mag=(0,3), direction=1, fwhm=0, std=std)[0]
  ramp_100_smth = ramp_2D(dim=dim_100, mag=(0,3), direction=1, fwhm=3*2, std=std)[0]

  fig, axs = plt.subplots(2, 4, figsize=figsize)

  im = axs[0, 0].imshow(fdr_cope(data=circular_l100, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[0, 0].set_title("large circle (100*100)", fontsize = fontsize)

  im = axs[0, 1].imshow(fdr_cope(data=circular_s100, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[0, 1].set_title("small circle (100*100)", fontsize = fontsize)

  im = axs[0, 2].imshow(fdr_cope(data=circular_l100_smth, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[0, 2].set_title("large circle (100*100, smoothed noise)", fontsize = fontsize)

  im = axs[0, 3].imshow(fdr_cope(data=circular_s100_smth, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[0, 3].set_title("small circle (100*100, smoothed noise)", fontsize = fontsize)

  im = axs[1, 0].imshow(fdr_cope(data=ramp_100, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[1, 0].set_title("ramp (100*100)", fontsize = fontsize)

  im = axs[1, 2].imshow(fdr_cope(data=ramp_100_smth, method=method, threshold=c, tail=tail)[3], vmin=_min, vmax=_max)
  axs[1, 2].set_title("ramp (100*100, smoothed noise)", fontsize = fontsize)

  axs[1, 1].text(0.5, 0.5, s='',
               fontsize = 20,horizontalalignment='center',
     verticalalignment='center')
  axs[1, 1].set_axis_off()
  axs[1, 3].text(0.5, 0.5, s='',
               fontsize = 20,horizontalalignment='center',
     verticalalignment='center')
  axs[1, 3].set_axis_off()

  cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.5])
  fig.colorbar(im, cax=cbar_ax)

  plt.show()