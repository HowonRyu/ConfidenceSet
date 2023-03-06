import numpy as np
import matplotlib.pyplot as plt

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