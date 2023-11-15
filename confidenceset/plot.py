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


# Error Check Plotting
def error_check_plot_single(sim_num, mode, shape, shape_spec, c, dim, ax, c_marg=0.2, alpha=0.05):
  """
  plots error rate simulation

  Parameters
  ----------
  sim_num : int
    simulation number
  mode : str
    options for error rate "FDR" or "FNDR"
  shape : str
    "ramp" or "ellipse"
  shape_spec : dict
    dictionary containing shape specs
  c : list
    list of thresholds
  dim : int
    dimension of the image (N, W, H)
  ax : axes
    subplot figure to use
  c_marg : int
    margin allowed for the threshold
  tail : str
    "one" or "two"
  alpha : int
    [0, 1] alpha level

  Examples
  --------
  shapes = ["circular", "ellipse", "ramp"]
  # 50*50
  shape_specs_50 = shape_spec[0]
  fig, axs = plt.subplots(len(shape_specs_50), 3, figsize=figsize)
  for i in range(len(shape_specs_50)):
      for j, shape in enumerate(shapes):
          ax = axs[i, j]
          error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_50[i][j], c=c, dim=dim_50, ax=ax)
          ax.set_title(f"{shape}, dim={dim_50}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "fdr":
            ax.set_ylim([0, 0.02])
          elif mode == "fndr":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """

  tbl_joint = error_check_sim_table(sim_num=sim_num, mode=mode, method="joint",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha*2)
  tbl_separate_lower, tbl_separate_upper = error_check_sim_table(sim_num=sim_num, mode=mode, method="separate",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha)
  tbl_separate_avg = (tbl_separate_lower + tbl_separate_upper)/2


  joint = np.mean(tbl_joint, axis=1)
  separate_lower = np.mean(tbl_separate_lower, axis=1)
  separate_upper = np.mean(tbl_separate_upper, axis=1)
  separate_avg = np.mean(tbl_separate_avg, axis=1)

  ys = [joint, separate_lower, separate_upper, separate_avg]
  names = ['Joint', 'Separate(lower)', 'Separate(upper)', 'Separate(avg)']

  #m0/m
  #_, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
  #m = np.sum(mu>2)
  #m0 = list()
  #for thres in c:
  # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))


  for i, y in enumerate(ys):
    ax.plot(c, y, label=names[i])


def error_check_plot(sim_num, c, mode, shape_spec, c_marg=0.2, tail="two", alpha=0.05, alpha0=0.05/4, alpha1=0.05/2, figsize=(15,10)):
  """
  combines error_check_plot_single to create a grid of simulations plots with different simulation settings

  Parameters
  ----------
  sim_num : int
    simulation number
  c : list
    list of thresholds
  mode : str
    options for error rate "FDR" or "FNDR"
  shape_spec : dict
    dictionary containing shape specs
  figsize : tuple
    figure size

  Examples
  --------
  error_check_plot(sim_num=100, mode="fdr", c=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], shape_spec=shape_specs_sim, figsize=(23,30))

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """

  shapes = ["circular", "ellipse", "ramp"]
  dim_50 = (80,50,50)
  dim_100 = (80,100,100)
  # 50*50
  shape_specs_50 = shape_spec[0]
  fig, axs = plt.subplots(len(shape_specs_50), 3, figsize=figsize)
  for i in range(len(shape_specs_50)):
      for j, shape in enumerate(shapes):
          ax = axs[i, j]
          error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape,  shape_spec=shape_specs_50[i][j], c=c,
                                  dim=dim_50, ax=ax, c_marg=c_marg, alpha=alpha)
          ax.set_title(f"{shape}, dim={dim_50}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "FDR":
            ax.set_ylim([0, 0.07])
          elif mode == "FNDR":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


  # 100*100
  #shape_specs_100 = shape_spec[1]
  #fig, axs = plt.subplots(len(shape_specs_100), 3, figsize=figsize)
  #for i in range(len(shape_specs_100)):
  #    for j, shape in enumerate(shapes):
  #        ax = axs[i, j]
  #        error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_100[i][j], c=c,
  #                                dim=dim_100, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
  #        ax.set_title(f"{shape}, dim={dim_100}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
  #        ax.set_xlabel("threshold")
  #        ax.set_ylabel(str(mode))
  #        if mode == "fdr":
  #          ax.set_ylim([0, 0.02])
  #        elif mode == "fndr":
  #          ax.set_ylim([0,1])
  #        ax.legend()
  #plt.show()