import numpy as np
import scipy.stats

def fdr_cope(data, threshold, alpha=0.05, tail="two"):
  """ 
  sub-setting the confidence set controlling for FDR

  Parameters
  ----------
  data : int
    array of voxels
  threshold : int
    threshold to be used for sub-setting
  alpha : int
    significance level
    

  Returns
  -------
  lower_set : Boolean
    voxels denoting the lower confidence set
  upper_set : Boolean
    voxels denoting the upper confidence set


  Example
  -------
  nsub = 50
  data = numpy.random.randn(nsub, 100, 100) + 2
  lower, upper = fdr_cope(data, threshold=2, alpha=0.05, tail="two)
  plt.imshow(lower)
  plt.imshow(upper)

  :Authors:
    Samuel Davenport <sdavenport@health.ucsd.edu>
    Howon Ryu <howonryu@ucsd.edu>
  """
  data_tstat = mvtstat(data - threshold)
  data_dim = data.shape
  nsubj = data_dim[0]
  Achat = data_tstat >= 0
  Achat_C = data_tstat < 0

  if tail == "two":
    pvals = 2*(1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1))

    rejection_ind, _, n_rej = fdrBH(pvals, alpha)
    outer_set = 1- Achat_C*rejection_ind
    inner_set = Achat*rejection_ind

  if tail == "one":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)

    inner_rejection_ind, _, n_rej = fdrBH(inner_pvals, alpha)
    outer_rejection_ind, _, n_rej = fdrBH(outer_pvals, alpha)
    outer_set = 1- Achat_C*outer_rejection_ind
    inner_set = Achat*inner_rejection_ind

  return(outer_set, inner_set, Achat, outer_set + inner_set + Achat, n_rej)




def fdrBH(pvalues, alpha=0.05):
  """ 
  running the Benjamini-Hochberg procedure for false discovery rate control

  Parameters
  ----------
  pvalues : int
    an array or list of p-values
  alpha : int
    significance level

  Returns
  -------
  rejection_ind : Boolean
    shows whether or not the voxel is rejected
  rejection_locs : int
    locations of the voxels that are rejected (flattened)
  nrejections : int
    number of voxels rejected

  Examples
  --------
  BHpvals = fdrBH([0.052, 0.02, 0.034, 0.05], alpha=0.05)

  data = numpy.random.randn(100,50,50)
  data_tstat = mvtstat(data - threshold)
  data_dim = data.shape
  nsubj = data_dim[0]
  pvals = 2*(1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1));
  rejection_ind, _, _ = fdrBH(pvals, alpha)
  """

  pvals_dim = pvalues.shape
  pvalues_flat = pvalues.flatten()
  sorted_pvalues = np.sort(pvalues_flat)
  sort_index = np.argsort(pvalues_flat)

  npvals = len(pvalues_flat)
  BH_upper = ((np.arange(npvals)+1)/npvals)*alpha #critical values from BH
  BH_vector = sorted_pvalues <= BH_upper

  if np.where(BH_vector)[0].size == 0:
    nrejections = 0
    rejection_locs = None  # flattened or None
    rejection_ind = np.full(np.prod(pvals_dim), 0).reshape(pvals_dim)

  else:
    nrejections = np.where(BH_vector)[-1][-1] + 1
    rejection_locs = np.sort(sort_index[0:nrejections])  # flattened
    rejection_ind = np.full(np.prod(pvals_dim), 0)
    rejection_ind[rejection_locs] = 1
    rejection_ind = rejection_ind.reshape(pvals_dim)




  
  return(rejection_ind, rejection_locs, nrejections)




def mvtstat(data):
  """ returns multivariate t-statistics
  
  Parameters
  ----------
  data : int
    array or list with first axis as the sample size

  Returns
  -------
  tstat : int
    t-statistics
  
  Example
  -------
  data = 5 + np.random.randn(100, 100)
  tstats = mvtstat(data)
  """
  dat_dim = data.shape
  img_dim = dat_dim[1:]
  img_dim_len = len(img_dim)
  n_subj = dat_dim[0]
  
  xbar = np.mean(data, axis=0)
  sq_xbar = np.mean(data**2, axis=0)
    
  est_var = (n_subj/(n_subj-1))*(sq_xbar - (xbar**2))
  std_dev = np.sqrt(est_var)

  tstat = (np.sqrt(n_subj) * xbar)/std_dev
  cohensd = xbar/std_dev

  return(tstat)



def conf_plot(mu_set, noise_set, field_dim, threshold, alpha=0.05):
  noise = get_noise(noise_set, np.array(field_dim))
  mu = get_mu(mu_set, np.array(field_dim))
  data = noise + mu
  outer, inner, Achat, nrej = fdr_cope(data, threshold=threshold, alpha=alpha, tail="two")
  plot = outer + inner + (mu[0,:,:]==threshold)
  return(plot)



def conf_plot_agg(c, _min=0, _max=3):
  fig, axs = plt.subplots(2, 4, figsize=(30, 20))

  im = axs[0, 0].imshow(conf_plot(mu_spec_circle_l_50, noise_spec_homogen, field_dim=(80,50,50),
          threshold=c, alpha=0.05), vmin=_min, vmax=_max)
  axs[0, 0].set_title("large circle (50*50)")


  im = axs[0, 1].imshow(conf_plot(mu_spec_circle_s_50, noise_spec_homogen, field_dim=(80,50,50),
          threshold=c, alpha=0.05), vmin=_min, vmax=_max)
  axs[0, 1].set_title("small circle (50*50)")

  im = axs[0, 2].imshow(conf_plot(mu_spec_circle_l_100, noise_spec_homogen, field_dim=(80,100,100),
          threshold=c, alpha=0.05), vmin=_min, vmax=_max)
  axs[0, 2].set_title("large circle (100*100)")

  im = axs[0, 3].imshow(conf_plot(mu_spec_circle_s_100, noise_spec_homogen, field_dim=(80,100,100),
          threshold=c, alpha=0.05), vmin=_min, vmax=_max)
  axs[0, 3].set_title("small circle (100*100)")


  im = axs[1, 0].imshow(conf_plot(mu_spec_circle_l_50, noise_spec_homogen_smth, field_dim=(80,50,50),
          threshold=c,  alpha=0.05), vmin=_min, vmax=_max)
  axs[1, 0].set_title("large circle (50*50, smoothed noise)")


  im = axs[1, 1].imshow(conf_plot(mu_spec_circle_s_50, noise_spec_homogen_smth, field_dim=(80,50,50),
          threshold=c, alpha=0.05),  vmin=_min, vmax=_max)
  axs[1, 1].set_title("small circle (50*50, smoothed noise)")

  im = axs[1, 2].imshow(conf_plot(mu_spec_circle_l_100, noise_spec_homogen_smth, field_dim=(80,100,100),
          threshold=c, alpha=0.05),  vmin=_min, vmax=_max, )
  axs[1, 2].set_title("large circle (100*100, smoothed noise)")


  im = axs[1, 3].imshow(conf_plot(mu_spec_circle_s_100, noise_spec_homogen_smth, field_dim=(80,100,100),
          threshold=c, alpha=0.05), vmin=_min, vmax=_max)
  axs[1, 3].set_title("small circle (100*100, smoothed noise)")

  cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.5])
  fig.colorbar(im, cax=cbar_ax)

  plt.show()




def fdr_error_check(n_subj, img_dim, c, noise_set, mu_set,
                    var=1, alpha=0.05, tail="two"):
    data_dim = np.array((n_subj,) + img_dim)
    noise = get_noise(noise_set, data_dim) * var
    mu = get_mu(mu_set, data_dim)
    data = mu + noise

    lower, upper, Achat, all_sets, _ = fdr_cope(data, threshold=c, alpha=0.05, tail=tail)
    Ac = mu >= c
    AcC = 1 - Ac
    upper_inclusion = 1 - np.any(AcC * upper > 0)
    lower_inclusion = 1 - np.any(Ac * (1 - lower) > 0)
    inclusion = upper_inclusion * lower_inclusion
    exclusion = 1 - inclusion
    return (exclusion)


def fdr_error_check2(n_subj, img_dim, c, noise_set, mu_set,
                     var=1, alpha=0.05, tail="two"):
    data_dim = np.array((n_subj,) + img_dim)
    noise = get_noise(noise_set, data_dim) * var
    mu = get_mu(mu_set, data_dim)
    data = mu + noise

    lower, upper, Achat, all_sets, n_rej = fdr_cope(data, threshold=c, alpha=0.05, tail=tail)
    Ac = mu >= c
    AcC = 1 - Ac

    numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
    denom = np.sum(np.minimum(upper + (1 - lower), 1))

    if n_rej == 0:
        ERR = 0
    else:
        ERR = numer / denom
    return (ERR)
