import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



### procedures

def fdr_cope(data, threshold, method, alpha=0.05, tail="two",
             k=2, alpha0=0.05/4, alpha1=0.05/2):
  """
  sub-setting the confidence set controlling for FDR

  Parameters
  ----------
  data : int
    array of voxels
  threshold : int
    threshold to be used for sub-setting
  alpha : int
    alpha level


  Returns
  -------
  lower_set : array(Boolean)
    voxels in the lower confidence set
  upper_set : array(Boolean)
    voxels in the upper confidence set
  Achat : Boolean
    voxels in the Ac_hat area
  plot_add : array(int)
    area representing lower_set + upper_set + Achat
  n_rej : int or list
    number of voxels rejected by the procedure

  Example
  -------
  nsub = 50
  data = numpy.random.randn(nsub, 100, 100) + 2
  lower, upper = fdr_cope(data, threshold=2, method="BH", alpha=0.05, tail="two)
  plt.imshow(lower)
  plt.imshow(upper)

  nsub = 50
  data = numpy.random.randn(nsub, 100, 100) + 2
  lower, upper = fdr_cope(data, threshold=2, method="AD", alpha0=0.05/4, alpha1 = 0.05/2, k=2 tail="two")
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
  n_rej = 0



  if tail == "two":
    pvals = 2 * (1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1))
    rejection_ind = np.full(np.prod(pvals.shape), 0)
    if method == "adaptive":
      rejection_ind, _, n_rej = fdr_adaptive(pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    if method == "BH":
      rejection_ind, _, n_rej = fdr_BH(pvals, alpha)
    outer_set = 1 - Achat_C * rejection_ind
    inner_set = Achat * rejection_ind

  if tail == "one":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    rejection_ind = np.full(np.prod(inner_pvals.shape), 0)
    if method == "adaptive":
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
      outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    if method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha)
      outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha)
    n_rej = [inner_n_rej, outer_n_rej]
    outer_set = 1 - Achat_C * outer_rejection_ind
    inner_set = Achat * inner_rejection_ind

  plot_add = outer_set + inner_set + Achat
  return(outer_set, inner_set, Achat, plot_add, n_rej)





def fdr_BH(pvalues, alpha=0.05):
  """
  the Benjamini-Hochberg procedure for false discovery rate control

  Parameters
  ----------
  pvalues : int
    an array or list of p-values
  alpha : int
    [0, 1] alpha level

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
  data = numpy.random.randn(100,50,50)
  data_tstat = mvtstat(data - threshold)
  data_dim = data.shape
  nsubj = data_dim[0]
  pvals = 2*(1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1));
  rejection_ind, _, _ = fdr_BH(pvals, alpha)

  :Authors:
    Samuel Davenport <sdavenport@health.ucsd.edu>
    Howon Ryu <howonryu@ucsd.edu>
  """
  pvalues = np.array(pvalues)
  pvals_dim = pvalues.shape
  pvalues_flat = pvalues.flatten()
  sorted_pvalues = np.sort(pvalues_flat)
  sort_index = np.argsort(pvalues_flat)

  m = len(pvalues_flat)
  delta_thres = ((np.arange(m) + 1) / m) * alpha  # threshold collection
  rejection = sorted_pvalues <= delta_thres

  if np.where(rejection)[0].size == 0:
    nrejections = 0
    rejection_locs = None  # flattened or None
    rejection_ind = np.full(np.prod(pvals_dim), 0).reshape(pvals_dim)

  else:
    nrejections = np.where(rejection)[-1][-1] + 1
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


def fdr_adaptive(pvalues, k, alpha0=0.05 / 4, alpha1=0.05 / 2):
  """
  the two-stage adaptive step-up procedure to control for false discovery rate

  Parameters
  ----------
  pvalues : int
    an array or list of p-values
  k : int
    parameter for the F_kap function
  alpha0 : int
    [0, 1] alpha level for the first stage
  alpha1 : int
    [0, 1] alpha level for the second stage

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
  data = numpy.random.randn(100,50,50)
  data_tstat = mvtstat(data - threshold)
  data_dim = data.shape
  nsubj = data_dim[0]
  pvals = 2*(1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1));
  rejection_ind, _, _ = fdr_adaptive(pvals, k=2, alpha0=0.05/4, alpha1=0.05/2)

  :Authors:
    Samuel Davenport <sdavenport@health.ucsd.edu>
    Howon Ryu <howonryu@ucsd.edu>
  """
  pvalues = np.array(pvalues)
  first_rejection_ind, first_rejection_locs, R0 = fdr_BH(pvalues, alpha=alpha0)
  pvals_dim = pvalues.shape
  pvalues_flat = pvalues.flatten()
  sorted_pvalues = np.sort(pvalues_flat)
  sort_index = np.argsort(pvalues_flat)

  m = len(pvalues_flat)
  delta_thres = (((np.arange(m) + 1) * F_kap(x=(R0 / m), k=k)) / m) * alpha1  # threshold collection
  rejection = sorted_pvalues <= delta_thres

  if np.where(rejection)[0].size == 0:
    nrejections = 0
    rejection_locs = None  # flattened or None
    rejection_ind = np.full(np.prod(pvals_dim), 0).reshape(pvals_dim)

  else:
    nrejections = np.where(rejection)[-1][-1] + 1
    rejection_locs = np.sort(sort_index[0:nrejections])  # flattened
    rejection_ind = np.full(np.prod(pvals_dim), 0)
    rejection_ind[rejection_locs] = 1
    rejection_ind = rejection_ind.reshape(pvals_dim)

  return(rejection_ind, rejection_locs, nrejections)





def F_kap(x, k):
  """
  F_k function

  Parameters
  ----------
  x : int
    input value for the function
  k : int
    parameter for the function

  Returns
  -------
  y : int
    output value from the function

  Examples
  --------
  pvalues = [0.005, 0.003, 0.006, 0.994, 0.002, 0.0001, 0.035]
  _, _, R0 = fdr_BH(pvalues, alpha=alpha0)
  m = len(pvalues_flat)
  print(F_kap(x=R0/m, k=2))

  :Authors:
    Samuel Davenport <sdavenport@health.ucsd.edu>
    Howon Ryu <howonryu@ucsd.edu>
  """
  if k < 2:
    return ("invalid k value")
  else:
    kinv = k ** (-1)
    if x > 1 or x < 0:
      return ("invalid X values")
    else:
      if x <= kinv:
        y = 1
      else:
        y = (2 * kinv) / (1 - np.sqrt(1 - 4 * (1 - x) * kinv))
  return(y)



### FDR and power tests

def fwe_inclusion_check(n_subj, img_dim, c, noise_set, mu_set, var=1, alpha=0.05, tail="two"):
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


def fdr_error_check(dim, c, shape, method, shape_spec=None, mag=3, direction=1, fwhm=3,
                    std=5, alpha=0.05, tail="two"):
  if (shape == 'noise'):
    data = np.random.randn(*dim) * std
    mu = np.zeros((dim[1], dim[2]))

  if (shape == "circular"):
    data, mu = circular_2D(dim=dim, shape_spec=shape_spec)
  if (shape == "ramp"):
    data, mu = ramp_2D(dim=dim, std=std, mag=(0, mag), direction=direction, fwhm=fwhm)

  Ac = mu >= c
  AcC = 1 - Ac
  lower, upper, Achat, all_sets, n_rej = fdr_cope(data, method=method, threshold=c, alpha=alpha, tail=tail)

  if tail == "one":
    numer = np.sum(np.maximum(upper - Ac.astype(int), 0))
    denom = np.sum(upper)

  if tail == "two":
    numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
    denom = np.sum(np.minimum(upper + (1 - lower), 1))

  if n_rej == 0:
    ERR = 0
  else:
    ERR = numer / denom

  return (ERR)


def fdr_error_check_sim(sim_num, method, c, c_marg=0.2, std=5, tail="two", alpha=0.05, alpha0=0.05 / 4,
                        alpha1=0.05 / 2):
  dim_100 = (80, 100, 100)
  dimprod_100 = dim_100[1] * dim_100[2]
  dim_50 = (80, 50, 50)
  dimprod_50 = dim_50[1] * dim_50[2]
  up0, lo0 = c[0] + c_marg, c[0] - c_marg
  up1, lo1 = c[1] + c_marg, c[1] - c_marg
  up2, lo2 = c[2] + c_marg, c[2] - c_marg

  # ramps
  mu_ramp_50 = ramp_2D(dim=dim_50, mag=(0, 3), direction=1, fwhm=0, std=std)[1]
  mu_ramp_100 = ramp_2D(dim=dim_100, mag=(0, 3), direction=1, fwhm=0, std=std)[1]

  m0_ramp_50_c0 = np.sum(np.logical_and(mu_ramp_50 < up0, mu_ramp_50 > lo0))
  m0_ramp_50_c1 = np.sum(np.logical_and(mu_ramp_50 < up1, mu_ramp_50 > lo1))
  m0_ramp_50_c2 = np.sum(np.logical_and(mu_ramp_50 < up2, mu_ramp_50 > lo2))
  m0_ramp_100_c0 = np.sum(np.logical_and(mu_ramp_100 < up0, mu_ramp_100 > lo0))
  m0_ramp_100_c1 = np.sum(np.logical_and(mu_ramp_100 < up1, mu_ramp_100 > lo1))
  m0_ramp_100_c2 = np.sum(np.logical_and(mu_ramp_100 < up2, mu_ramp_100 > lo2))

  # circles
  spec_cir_s50 = {'r': 0.45, 'std': std, 'mag': 3, 'fwhm_noise': 0, 'fwhm_signal': 10}
  spec_cir_s50_smth = {'r': 0.45, 'std': std, 'mag': 3, 'fwhm_noise': 3, 'fwhm_signal': 10}
  spec_cir_l50 = {'r': 0.8, 'std': std, 'mag': 3, 'fwhm_noise': 0, 'fwhm_signal': 10}
  spec_cir_l50_smth = {'r': 0.8, 'std': std, 'mag': 3, 'fwhm_noise': 3, 'fwhm_signal': 10}

  spec_cir_s100 = {'r': 0.45, 'std': std, 'mag': 3, 'fwhm_noise': 0, 'fwhm_signal': 10 * 2}
  spec_cir_s100_smth = {'r': 0.45, 'std': std, 'mag': 3, 'fwhm_noise': 3 * 2, 'fwhm_signal': 10 * 2}
  spec_cir_l100 = {'r': 0.8, 'std': std, 'mag': 3, 'fwhm_noise': 0, 'fwhm_signal': 10 * 2}
  spec_cir_l100_smth = {'r': 0.8, 'std': std, 'mag': 3, 'fwhm_noise': 3 * 2, 'fwhm_signal': 10 * 2}

  mu_circular_s50 = circular_2D(dim=dim_50, shape_spec=spec_cir_s50)[1]
  mu_circular_l50 = circular_2D(dim=dim_50, shape_spec=spec_cir_l50)[1]
  m0_small50_c0 = np.sum(np.logical_and(mu_circular_s50 < up0, mu_circular_s50 > lo0))
  m0_small50_c1 = np.sum(np.logical_and(mu_circular_s50 < up1, mu_circular_s50 > lo1))
  m0_small50_c2 = np.sum(np.logical_and(mu_circular_s50 < up2, mu_circular_s50 > lo2))
  m0_large50_c0 = np.sum(np.logical_and(mu_circular_l50 < up0, mu_circular_l50 > lo0))
  m0_large50_c1 = np.sum(np.logical_and(mu_circular_l50 < up1, mu_circular_l50 > lo1))
  m0_large50_c2 = np.sum(np.logical_and(mu_circular_l50 < up2, mu_circular_l50 > lo2))

  mu_circular_s100 = circular_2D(dim=dim_100, shape_spec=spec_cir_s100)[1]
  mu_circular_l100 = circular_2D(dim=dim_100, shape_spec=spec_cir_l100)[1]
  m0_small100_c0 = np.sum(np.logical_and(mu_circular_s100 < up0, mu_circular_s100 > lo0))
  m0_small100_c1 = np.sum(np.logical_and(mu_circular_s100 < up1, mu_circular_s100 > lo1))
  m0_small100_c2 = np.sum(np.logical_and(mu_circular_s100 < up2, mu_circular_s100 > lo2))
  m0_large100_c0 = np.sum(np.logical_and(mu_circular_l100 < up0, mu_circular_l100 > lo0))
  m0_large100_c1 = np.sum(np.logical_and(mu_circular_l100 < up1, mu_circular_l100 > lo1))
  m0_large100_c2 = np.sum(np.logical_and(mu_circular_l100 < up2, mu_circular_l100 > lo2))

  ###################### 50*50 #########################
  # initializing and labeling
  ERR = dict()
  ERR['threshold'] = ["c=" + str(c[0]), "c=" + str(c[1]), "c=" + str(c[2])]

  # small 50*50
  ERR['circle(s)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle(s)'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_s50, alpha=alpha,
                      tail=tail))
    ERR['circle(s)'][1].append(
      fdr_error_check(dim_50, c=c[1], shape="circular", method=method, shape_spec=spec_cir_s50, alpha=alpha,
                      tail=tail))
    ERR['circle(s)'][2].append(
      fdr_error_check(dim_50, c=c[2], shape="circular", method=method, shape_spec=spec_cir_s50, alpha=alpha,
                      tail=tail))

  # small_smth 50*50
  ERR['circle(s)smth'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle(s)smth'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_s50_smth, alpha=alpha,
                      tail=tail))
    ERR['circle(s)smth'][1].append(
      fdr_error_check(dim_50, c=c[1], shape="circular", method=method, shape_spec=spec_cir_s50_smth, alpha=alpha,
                      tail=tail))
    ERR['circle(s)smth'][2].append(
      fdr_error_check(dim_50, c=c[2], shape="circular", method=method, shape_spec=spec_cir_s50_smth, alpha=alpha,
                      tail=tail))

  if method == "adaptive":
    ERR['alpha(small)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR['alpha*m0/m(small)'] = [np.round(0.05 * m0_small50_c0 / (dimprod_50), 5),
                                np.round(0.05 * m0_small50_c1 / (dimprod_50), 5),
                                np.round(0.05 * m0_small50_c2 / (dimprod_50), 5)]


  # large 50*50
  ERR['circle(l)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle(l)'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l50, alpha=alpha,
                      tail=tail))
    ERR['circle(l)'][1].append(
      fdr_error_check(dim_50, c=c[1], shape="circular", method=method, shape_spec=spec_cir_l50, alpha=alpha,
                      tail=tail))
    ERR['circle(l)'][2].append(
      fdr_error_check(dim_50, c=c[2], shape="circular", method=method, shape_spec=spec_cir_l50, alpha=alpha,
                      tail=tail))

  # large_smth 50*50
  ERR['circle(l)smth'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR['circle(l)smth'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l50_smth, alpha=alpha,
                      tail=tail))
    ERR['circle(l)smth'][1].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l50_smth, alpha=alpha,
                      tail=tail))
    ERR['circle(l)smth'][2].append(
      fdr_error_check(dim_50, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l50_smth, alpha=alpha,
                      tail=tail))


  if method == "adaptive":
    ERR['alpha(large)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR['alpha*m0/m(large)'] = [np.round(0.05 * m0_large50_c0 / (dimprod_50), 5),
                                np.round(0.05 * m0_large50_c1 / (dimprod_50), 5),
                                np.round(0.05 * m0_large50_c2 / (dimprod_50), 5)]

  # ramp 50*50
  ERR['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))
    ERR['ramp'][1].append(
      fdr_error_check(dim_50, c=c[1], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))
    ERR['ramp'][2].append(
      fdr_error_check(dim_50, c=c[2], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))

  # ramp_smth 50*50
  ERR['ramp_smth'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp_smth'][0].append(
      fdr_error_check(dim_50, c=c[0], shape="ramp", method=method, mag=3, direction=1, fwhm=3, std=std,
                      alpha=alpha, tail=tail))
    ERR['ramp_smth'][1].append(
      fdr_error_check(dim_50, c=c[1], shape="ramp", method=method, mag=3, direction=1, fwhm=3, std=std,
                      alpha=alpha, tail=tail))
    ERR['ramp_smth'][2].append(
      fdr_error_check(dim_50, c=c[2], shape="ramp", method=method, mag=3, direction=1, fwhm=3, std=std,
                      alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR['alpha(ramp)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR['alpha*m0/m(ramp)'] = [np.round(0.05 * m0_ramp_50_c0 / (dimprod_50), 5),
                               np.round(0.05 * m0_ramp_50_c1 / (dimprod_50), 5),
                               np.round(0.05 * m0_ramp_50_c2 / (dimprod_50), 5)]


  ERR_key_calc = [list(ERR.keys())[i] for i in [1, 2, 4, 5, 7, 8]]
  ERR.update({n: np.round(np.nanmean(ERR[n], axis=1), 4) for n in ERR_key_calc})

  ###################### 100*100 #########################
  # initializing and labeling
  ERR2 = dict()
  ERR2['threshold'] = ["c=" + str(c[0]), "c=" + str(c[1]), "c=" + str(c[2])]

  # small 100*100
  ERR2['circle(s)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle(s)'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_s100, alpha=alpha,
                      tail=tail))
    ERR2['circle(s)'][1].append(
      fdr_error_check(dim_100, c=c[1], shape="circular", method=method, shape_spec=spec_cir_s100, alpha=alpha,
                      tail=tail))
    ERR2['circle(s)'][2].append(
      fdr_error_check(dim_100, c=c[2], shape="circular", method=method, shape_spec=spec_cir_s100, alpha=alpha,
                      tail=tail))

  # small_smth 100*100
  ERR2['circle(s)smth'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle(s)smth'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_s100_smth,
                      alpha=alpha, tail=tail))
    ERR2['circle(s)smth'][1].append(
      fdr_error_check(dim_100, c=c[1], shape="circular", method=method, shape_spec=spec_cir_s100_smth,
                      alpha=alpha, tail=tail))
    ERR2['circle(s)smth'][2].append(
      fdr_error_check(dim_100, c=c[2], shape="circular", method=method, shape_spec=spec_cir_s100_smth,
                      alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['alpha(small)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(small)'] = [np.round(0.05 * m0_small100_c0 / (dimprod_100), 5),
                                 np.round(0.05 * m0_small100_c1 / (dimprod_100), 5),
                                 np.round(0.05 * m0_small100_c2 / (dimprod_100), 5)]

  # large 100*100
  ERR2['circle(l)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle(l)'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l100, alpha=alpha,
                      tail=tail))
    ERR2['circle(l)'][1].append(
      fdr_error_check(dim_100, c=c[1], shape="circular", method=method, shape_spec=spec_cir_l100, alpha=alpha,
                      tail=tail))
    ERR2['circle(l)'][2].append(
      fdr_error_check(dim_100, c=c[2], shape="circular", method=method, shape_spec=spec_cir_l100, alpha=alpha,
                      tail=tail))

  # large_smth 100*100
  ERR2['circle(l)smth'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR2['circle(l)smth'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l100_smth,
                      alpha=alpha, tail=tail))
    ERR2['circle(l)smth'][1].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l100_smth,
                      alpha=alpha, tail=tail))
    ERR2['circle(l)smth'][2].append(
      fdr_error_check(dim_100, c=c[0], shape="circular", method=method, shape_spec=spec_cir_l100_smth,
                      alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['alpha(large)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(large)'] = [np.round(0.05 * m0_large100_c0 / (dimprod_100), 5),
                                 np.round(0.05 * m0_large100_c1 / (dimprod_100), 5),
                                 np.round(0.05 * m0_large100_c2 / (dimprod_100), 5)]


  # ramp 100*100
  ERR2['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))
    ERR2['ramp'][1].append(
      fdr_error_check(dim_100, c=c[1], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))
    ERR2['ramp'][2].append(
      fdr_error_check(dim_100, c=c[2], shape="ramp", method=method, mag=3, direction=1, fwhm=0, std=std,
                      alpha=alpha, tail=tail))

  # ramp_smth 50*50
  ERR2['ramp_smth'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp_smth'][0].append(
      fdr_error_check(dim_100, c=c[0], shape="ramp", method=method, mag=3, direction=1, fwhm=3 * 2, std=std,
                      alpha=alpha, tail=tail))
    ERR2['ramp_smth'][1].append(
      fdr_error_check(dim_100, c=c[1], shape="ramp", method=method, mag=3, direction=1, fwhm=3 * 2, std=std,
                      alpha=alpha, tail=tail))
    ERR2['ramp_smth'][2].append(
      fdr_error_check(dim_100, c=c[2], shape="ramp", method=method, mag=3, direction=1, fwhm=3 * 2, std=std,
                      alpha=alpha, tail=tail))


  if method == "adaptive":
    ERR2['alpha(large)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(ramp)'] = [np.round(0.05 * m0_ramp_100_c0 / (100 * 100), 5),
                                np.round(0.05 * m0_ramp_100_c1 / (100 * 100), 5),
                                np.round(0.05 * m0_ramp_100_c2 / (100 * 100), 5)]

  ERR2_key_calc = [list(ERR2.keys())[i] for i in [1, 2, 4, 5, 7, 8]]
  ERR2.update({n: np.round(np.nanmean(ERR2[n], axis=1), 4) for n in ERR2_key_calc})

  return (ERR, ERR2)




### random field generator
def gen_2D(dim, shape, shape_spec, truncate=4):
  fwhm_noise = shape_spec['fwhm_noise']
  std = shape_spec['std']

  # signal
  if shape == "ramp":
    signal = ramp_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  if shape == "ellipse":
    signal = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)

  # noise
  noise = np.random.randn(*dim) * std
  sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma_noise, truncate=truncate)  #smoothing

  data = np.array(mu + noise, dtype='float')
  return(data, signal)

def ramp_2D(dim, shape_spec):
  nsubj = dim[0]
  direction = shape_spec['direction']
  mag = shape_spec['mag']
  std = shape_spec['std']

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  else: #horizontal
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2]),dim[1]).reshape(dim[2],dim[1]).transpose()
  mu = np.array(mu_temp, dtype='float')
  return(mu)

def ellipse_2D(dim, shape_spec, truncate=4):
  nsubj = dim[0]
  a = shape_spec['a']
  b = shape_spec['b']
  mag = shape_spec['mag']
  fwhm_signal = shape_spec['fwhm_signal']

  # signal
  x, y = np.meshgrid(np.linspace(-1,1,dim[1]), np.linspace(-1,1,dim[2]))
  cx, cy = 0,0
  theta = -np.pi/4
  xx = np.cos(theta)*(x-cx) + np.sin(theta)*(y-cy)
  yy = -np.sin(theta)*(x-cx) + np.cos(theta)*(y-cy)
  ellipse = np.array((xx/a)**2 + (yy/b)**2 <= 1, dtype="float")

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  ellipse_smth = gaussian_filter(ellipse, sigma = sigma_signal, truncate=truncate)
  mu = np.array(ellipse_smth * mag, dtype='float')

  return(mu)


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