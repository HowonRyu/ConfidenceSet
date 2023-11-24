import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
from .test import *


# Confidence Set Plotting
def conf_plot_agg(threshold, method, seed=None, r=0.5, std = 5, fwhm_noise=3,
                  fwhm_signal=20, mag = 3, fontsize = 25, figsize=(30, 20),
                   alpha=0.05, k=2, alpha0=0.05 / 4, alpha1=0.05 / 2):

  """
  plots FDR controlling confidence sets for six different random fields

  Parameters
  ----------
    threshold : int
      threshold c
    method : str
      "separate" or "joint"
    r : int
      radii of ellipses
    std : int
      standard deviation for the noise field N(0, std^2)
    mag : int
      magnitude of the signal
    fontsize : int
      font size for figure
    figsize : tuple
      figure size
    k : int
      kappa level for the adaptive procedure
    alpha0 : int
      [0,1] alpha0 level for the adaptive procedure
    alpha1 : int
      [0,1] alpha1 level for the adaptive procedure

  Examples
  --------
  conf_plot_agg(threshold=2, method = "joint", std=7, figsize = (5,3))

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """

  # setting up colormap
  cmap1 = colors.ListedColormap(['black', 'blue'])
  cmap2 = colors.ListedColormap(['none', 'yellow'])
  cmap22 = colors.ListedColormap(['black', 'yellow'])
  cmap3 = colors.ListedColormap(['none', 'red'])

  # setting up data specs and generating field
  dim_100 = (80,100,100)
  spec_cir_100_smth = {'a':r, 'b':r, 'std':std, 'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_elp_100_smth = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_ramp_100_smth = {'direction':1, 'std':std, 'mag':(0,mag), 'fwhm_noise':fwhm_noise}
  circular_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_cir_100_smth)
  ellipse_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_elp_100_smth)
  ramp_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ramp", shape_spec=spec_ramp_100_smth)


  # plotting
  cmap = colors.ListedColormap(['black', 'blue', 'yellow', 'red'])
  fig, axs = plt.subplots(1,3, figsize=figsize)

  im = axs[0].imshow(fdr_confset(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[0].imshow(fdr_confset(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  im = axs[0].imshow(fdr_confset(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  axs[0].set_title("circle", fontsize = fontsize)

  im = axs[1].imshow(fdr_confset(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[1].imshow(fdr_confset(data=ellipse_100_smth, method=method, alpha=alpha,threshold=threshold)[2], cmap=cmap2)
  im = axs[1].imshow(fdr_confset(data=ellipse_100_smth, method=method, alpha=alpha,threshold=threshold)[1], cmap=cmap3)
  axs[1].set_title("ellipse", fontsize = fontsize)

  im = axs[2].imshow(fdr_confset(data=ramp_100_smth,  method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[2].imshow(fdr_confset(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  im = axs[2].imshow(fdr_confset(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  axs[2].set_title("ramp", fontsize = fontsize)

  plt.suptitle(f"{method} error control, alpha={alpha}")
  plt.show()

