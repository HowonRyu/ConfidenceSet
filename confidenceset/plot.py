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



# Simulation signal plotting

def get_sim_signal(shape, shape_spec, fwhm_signal_vec, fwhm_noise_vec):
  """
  produces plots of simulation signals (first row) and fields (signal + noise)

  Parameters
  ----------
  shape : str
    shape of the signal; choose from ramp or step. The rest is automatically ellipse.
  shape_spec : dict
    dictionary storing shape parameters
  fwhm_signal_vec : list
    fwhm values (float) for signal (columns)
  fwhm_noise_vec : list
    fwhm values (float) for noise (rows)

  Returns
  -------

  Example
  -------
  shape_spec_circle = {'a':0.5, 'b':0.5, 'mag':3,
                  'fwhm_signal': 0,
                  'fwhm_noise': 0,
                  'std': 1 }
  get_sim_signal(shape="circle", shape_spec=shape_spec_circle, fwhm_signal_vec=[5, 10, 15], fwhm_noise_vec=[0,5,10])

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  fig, axs = plt.subplots(len(fwhm_noise_vec)+1, len(fwhm_signal_vec), figsize=(15,20))


  for signal_idx, fwhm_signal in enumerate(fwhm_signal_vec):
    for noise_idx, fwhm_noise in enumerate(fwhm_noise_vec):
      ax = axs[noise_idx+1, signal_idx]
      shape_spec['fwhm_noise'] = fwhm_noise
      shape_spec['fwhm_signal'] = fwhm_signal
      data_sim, mu_sim = gen_2D(dim=(1, 50, 50), shape=shape, shape_spec=shape_spec, seed=12352, truncate=3)
      ax.set_title(f"fwhm_noise={fwhm_noise}, fwhm_signal={fwhm_signal}")
      img=ax.imshow(data_sim[0,:,:])
    ax = axs[0, signal_idx]
    ax.set_title(f"mu_signal, fwhm_signal={fwhm_signal}")
    img=ax.imshow(mu_sim)


  cbar_ax = fig.add_axes([0.95, 0.25, 0.015, 0.5])
  fig.colorbar(img, cax=cbar_ax, ticks=np.linspace(-1, 3, 9))
  plt.show()


def plot_confset_HCP(thresholds, background_slc, slc_info, muhat_file, sigma_file, resid_files, alpha, n_boot, misc, fontsize=15, figsize=[15,20]):
    cmap1 = colors.ListedColormap(['none', 'blue'])
    cmap2 = colors.ListedColormap(['none', 'yellow'])
    cmap3 = colors.ListedColormap(['none', 'red'])

    methods = ["joint", "separate_BH", "separate_adaptive", "SSS"]
    fig, axs = plt.subplots(len(thresholds), len(methods), figsize=figsize)
    for j, method in enumerate(methods):
        if j == "joint":
            alpha = 0.1
        else:
            alpha = 0.05

        for i, c in enumerate(thresholds):
            if method == "SSS":
                out_dir = "/Users/howonryu/Projects/ConfidenceSet/ConfidenceSet/SSS_output"
                upper_cr_file, lower_cr_file, estimated_ac_file, quantile_estimate = generate_CRs(muhat_file, sigma_file, resid_files, out_dir, c, 1-alpha, n_boot=n_boot)

                #upper_cr_file, lower_cr_file, estimated_ac_file, quantile_estimate = generate_CRs(mean_fname=muhat_file, sig_fname=sigma_file, 
                #res_fnames=resid_files, c=c, p=1-alpha, output=False, n_boot=n_boot)

                lower_set_all = nib.load(lower_cr_file).get_fdata()[:,:,:,0]
                upper_set_all = nib.load(upper_cr_file).get_fdata()[:,:,:,0]
                Achat_all = nib.load(estimated_ac_file).get_fdata()[:,:,:,0]
                if slc_info[1] == "sagittal":
                    axis="X"
                    lower_set = np.rot90(lower_set_all[slc_info[2],:,:], k=1)
                    upper_set = np.rot90(upper_set_all[slc_info[2],:,:], k=1)
                    Achat = np.rot90(Achat_all[slc_info[2],:,:], k=1)
                if slc_info[1] == "coronal":
                    axis="Y"
                    lower_set = np.rot90(lower_set_all[:,slc_info[2],:], k=1)
                    upper_set = np.rot90(upper_set_all[:,slc_info[2],:], k=1)
                    Achat = np.rot90(Achat_all[:,slc_info[2],:], k=1)
                if slc_info[1] == "axial":
                    axis="Z"
                    lower_set = np.rot90(lower_set_all[:,:,slc_info[2]], k=1)
                    upper_set = np.rot90(upper_set_all[:,:,slc_info[2]], k=1)
                    Achat = np.rot90(Achat_all[:,:,slc_info[2]], k=1)           
            else:
                lower_set, upper_set, Achat, plot_add, n_rej = fdr_confset(slc_info[0], threshold=c, method=method, alpha=alpha, k=2, alpha0=alpha / 4, alpha1=alpha / 2)

            # plot
            axs[i,j].imshow(background_slc, cmap="Greys_r")
            axs[i,j].imshow(lower_set, cmap=cmap1)
            axs[i,j].imshow(Achat, cmap=cmap2)
            axs[i,j].imshow(upper_set, cmap=cmap3)
            axs[i,j].axis('off')
            axs[i,j].set_title(f"threshold={c/10}% ({method})")

    plt.suptitle(f'{slc_info[1]} ({axis}={misc}), alpha = {alpha}', fontsize=fontsize)
        