import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors


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


def fdr_adaptive(pvalues, k=2, alpha0=0.05 / 4, alpha1=0.05 / 2):
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


def error_check(mode, dim, threshold, method, shape, shape_spec=None, alpha=0.05, tail="two"):
  if shape == 'noise':
    data = np.random.randn(*dim) * std
    mu = np.zeros((dim[1], dim[2]))

  else:
    data, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)

  Ac = mu >= threshold
  AcC = 1 - Ac
  lower, upper, Achat, all_sets, n_rej = fdr_cope(data, method=method, threshold=threshold, alpha=alpha, tail=tail)

  if mode == "fdr":
    if tail == "one":
      numer = np.sum(np.maximum(upper - Ac.astype(int), 0))
      denom = np.sum(upper)

    if tail == "two":
      numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
      denom = np.sum(1-np.minimum(lower - upper, 1))

  if mode == "fndr":
    if tail == "one":
      numer = np.sum(np.maximum(Ac.astype(int) - upper, 0))
      denom = np.sum(1-upper)

    if tail == "two":
      numer = np.sum(np.minimum(np.maximum(Ac.astype(int) - upper, 0) + np.maximum(lower - Ac.astype(int), 0), 1))
      denom = np.sum(np.minimum(lower - upper, 1))

  if mode == "type2":
    if tail == "one":
      numer = np.sum(np.maximum(Ac.astype(int) - upper, 0))
      denom = np.sum(Ac.astype(int))

    if tail == "two":
      numer = np.sum(np.minimum(np.maximum(Ac.astype(int) - upper, 0) + np.maximum(lower - Ac.astype(int), 0), 1))
      denom = np.sum(Ac.astype(int))


  if n_rej == 0:
    ERR = 0
  else:
    ERR = numer / denom

  return (ERR)


def error_check_sim(sim_num, mode, method, c, c_marg=0.2, std=5, tail="two", alpha=0.05, alpha0=0.05 / 4,
                    alpha1=0.05 / 2):
  dim_100 = (80, 100, 100)
  dimprod_100 = dim_100[1] * dim_100[2]
  dim_50 = (80, 50, 50)
  dimprod_50 = dim_50[1] * dim_50[2]
  up0, lo0 = c[0] + c_marg, c[0] - c_marg
  up1, lo1 = c[1] + c_marg, c[1] - c_marg
  up2, lo2 = c[2] + c_marg, c[2] - c_marg
  r=0.5
  mag = 3
  f50 = 10
  f100 = 10 * 2
  spec_cir_50 = {'a': r, 'b': r, 'std': std, 'mag': mag, 'fwhm_noise': 0, 'fwhm_signal': f50}
  spec_elp_50 = {'a': r * 2, 'b': r * 0.5, 'std': std, 'mag': mag, 'fwhm_noise': 0, 'fwhm_signal': f50}
  spec_ramp_50 = {'direction': 1, 'std': std, 'mag': (0, mag), 'fwhm_noise': 0}
  spec_cir_50_smth = {'a': r, 'b': r, 'std': std, 'mag': mag, 'fwhm_noise': (3 / 2), 'fwhm_signal': f50}
  spec_elp_50_smth = {'a': r * 2, 'b': r * 0.5, 'std': std, 'mag': mag, 'fwhm_noise': (3 / 2), 'fwhm_signal': f50}
  spec_ramp_50_smth = {'direction': 1, 'std': std, 'mag': (0, mag), 'fwhm_noise': (3 / 2)}

  spec_cir_100 = {'a': r, 'b': r, 'std': std, 'mag': mag, 'fwhm_noise': 0, 'fwhm_signal': f100}
  spec_elp_100 = {'a': r * 2, 'b': r * 0.5, 'std': std, 'mag': mag, 'fwhm_noise': 0, 'fwhm_signal': f100}
  spec_ramp_100 = {'direction': 1, 'std': std, 'mag': (0, mag), 'fwhm_noise': 0}
  spec_cir_100_smth = {'a': r, 'b': r, 'std': std, 'mag': mag, 'fwhm_noise': 3, 'fwhm_signal': f100}
  spec_elp_100_smth = {'a': r * 2, 'b': r * 0.5, 'std': std, 'mag': mag, 'fwhm_noise': 3, 'fwhm_signal': f100}
  spec_ramp_100_smth = {'direction': 1, 'std': std, 'mag': (0, mag), 'fwhm_noise': 3}

  # random field generator
  circular_50, mu_circular_50 = gen_2D(dim_50, shape="circular", shape_spec=spec_cir_50)
  ellipse_50, mu_ellipse_50 = gen_2D(dim_50, shape="ellipse", shape_spec=spec_elp_50)
  ramp_50, mu_ramp_50 = gen_2D(dim_50, shape="ramp", shape_spec=spec_ramp_50)
  circular_100, mu_circular_100 = gen_2D(dim_100, shape="circular", shape_spec=spec_cir_100)
  ellipse_100, mu_ellipse_100 = gen_2D(dim_100, shape="ellipse", shape_spec=spec_elp_100)
  ramp_100, mu_ramp_100 = gen_2D(dim_100, shape="ramp", shape_spec=spec_ramp_100)

  circular_50_smth, _ = gen_2D(dim_50, shape="circular", shape_spec=spec_cir_50_smth)
  ellipse_50_smth, _ = gen_2D(dim_50, shape="ellipse", shape_spec=spec_elp_50_smth)
  ramp_50_smth, _ = gen_2D(dim_50, shape="ramp", shape_spec=spec_ramp_50_smth)
  circular_100_smth, _ = gen_2D(dim_100, shape="circular", shape_spec=spec_cir_100_smth)
  ellipse_100_smth, _ = gen_2D(dim_100, shape="ellipse", shape_spec=spec_elp_100_smth)
  ramp_100_smth, _ = gen_2D(dim_100, shape="ramp", shape_spec=spec_ramp_100_smth)

  if c_marg == 0:
    m0_ramp_50_c0 = np.sum(mu_ramp_50 == c[0])
    m0_ramp_50_c1 = np.sum(mu_ramp_50 == c[1])
    m0_ramp_50_c2 = np.sum(mu_ramp_50 == c[2])
    m0_ramp_100_c0 = np.sum(mu_ramp_100 == c[0])
    m0_ramp_100_c1 = np.sum(mu_ramp_100 == c[1])
    m0_ramp_100_c2 = np.sum(mu_ramp_100 == c[2])
    # circle m0
    m0_circular_50_c0 = np.sum(mu_circular_50 == c[0])
    m0_circular_50_c1 = np.sum(mu_circular_50 == c[1])
    m0_circular_50_c2 = np.sum(mu_circular_50 == c[2])
    m0_circular_100_c0 = np.sum(mu_circular_100 == c[0])
    m0_circular_100_c1 = np.sum(mu_circular_100 == c[1])
    m0_circular_100_c2 = np.sum(mu_circular_100 == c[2])
    # ellipse m0
    m0_ellipse_50_c0 = np.sum(mu_ellipse_50 == c[0])
    m0_ellipse_50_c1 = np.sum(mu_ellipse_50 == c[1])
    m0_ellipse_50_c2 = np.sum(mu_ellipse_50 == c[2])
    m0_ellipse_100_c0 = np.sum(mu_ellipse_100 == c[0])
    m0_ellipse_100_c1 = np.sum(mu_ellipse_100 == c[1])
    m0_ellipse_100_c2 = np.sum(mu_ellipse_100 == c[2])

  else:
    # ramp m0
    m0_ramp_50_c0 = np.sum(np.logical_and(mu_ramp_50 < up0, mu_ramp_50 > lo0))
    m0_ramp_50_c1 = np.sum(np.logical_and(mu_ramp_50 < up1, mu_ramp_50 > lo1))
    m0_ramp_50_c2 = np.sum(np.logical_and(mu_ramp_50 < up2, mu_ramp_50 > lo2))
    m0_ramp_100_c0 = np.sum(np.logical_and(mu_ramp_100 < up0, mu_ramp_100 > lo0))
    m0_ramp_100_c1 = np.sum(np.logical_and(mu_ramp_100 < up1, mu_ramp_100 > lo1))
    m0_ramp_100_c2 = np.sum(np.logical_and(mu_ramp_100 < up2, mu_ramp_100 > lo2))
    # circle m0
    m0_circular_50_c0 = np.sum(np.logical_and(mu_circular_50 < up0, mu_circular_50 > lo0))
    m0_circular_50_c1 = np.sum(np.logical_and(mu_circular_50 < up1, mu_circular_50 > lo1))
    m0_circular_50_c2 = np.sum(np.logical_and(mu_circular_50 < up1, mu_circular_50 > lo1))
    m0_circular_100_c0 = np.sum(np.logical_and(mu_circular_100 < up0, mu_circular_100 > lo0))
    m0_circular_100_c1 = np.sum(np.logical_and(mu_circular_100 < up1, mu_circular_100 > lo1))
    m0_circular_100_c2 = np.sum(np.logical_and(mu_circular_100 < up1, mu_circular_100 > lo1))
    # ellipse m0
    m0_ellipse_50_c0 = np.sum(np.logical_and(mu_ellipse_50 < up0, mu_ellipse_50 > lo0))
    m0_ellipse_50_c1 = np.sum(np.logical_and(mu_ellipse_50 < up1, mu_ellipse_50 > lo1))
    m0_ellipse_50_c2 = np.sum(np.logical_and(mu_ellipse_50 < up2, mu_ellipse_50 > lo2))
    m0_ellipse_100_c0 = np.sum(np.logical_and(mu_ellipse_100 < up0, mu_ellipse_100 > lo0))
    m0_ellipse_100_c1 = np.sum(np.logical_and(mu_ellipse_100 < up1, mu_ellipse_100 > lo1))
    m0_ellipse_100_c2 = np.sum(np.logical_and(mu_ellipse_100 < up2, mu_ellipse_100 > lo2))

  ###################### 50*50 #########################
  # initializing and labeling
  ERR = dict()
  ERR['threshold'] = ["c=" + str(c[0]), "c=" + str(c[1]), "c=" + str(c[2])]

  # circle
  ERR['circle'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method, shape_spec=spec_cir_50,
                  alpha=alpha,
                  tail=tail))
    ERR['circle'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method, shape_spec=spec_cir_50,
                  alpha=alpha,
                  tail=tail))
    ERR['circle'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method, shape_spec=spec_cir_50,
                  alpha=alpha,
                  tail=tail))

  # circle smth
  ERR['circle(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle(smth)'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method, shape_spec=spec_cir_50_smth,
                  alpha=alpha,
                  tail=tail))
    ERR['circle(smth)'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method, shape_spec=spec_cir_50_smth,
                  alpha=alpha,
                  tail=tail))
    ERR['circle(smth)'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method, shape_spec=spec_cir_50_smth,
                  alpha=alpha,
                  tail=tail))

  if method == "adaptive":
    ERR['alpha(circle)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR['alpha*m0/m(circle)'] = [np.round(0.05 * m0_circular_50_c0 / (dimprod_50), 5),
                                 np.round(0.05 * m0_circular_50_c1 / (dimprod_50), 5),
                                 np.round(0.05 * m0_circular_50_c2 / (dimprod_50), 5)]

  # ellipse
  ERR['ellipse'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ellipse'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="ellipse", method=method, shape_spec=spec_elp_50,
                  alpha=alpha,
                  tail=tail))
    ERR['ellipse'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="ellipse", method=method, shape_spec=spec_elp_50,
                  alpha=alpha,
                  tail=tail))
    ERR['ellipse'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="ellipse", method=method, shape_spec=spec_elp_50,
                  alpha=alpha,
                  tail=tail))

  # ellipse smth
  ERR['ellipse(smth)'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR['ellipse(smth)'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method, shape_spec=spec_elp_50_smth,
                  alpha=alpha,
                  tail=tail))
    ERR['ellipse(smth)'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method, shape_spec=spec_elp_50_smth,
                  alpha=alpha,
                  tail=tail))
    ERR['ellipse(smth)'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method, shape_spec=spec_elp_50_smth,
                  alpha=alpha,
                  tail=tail))

  if method == "adaptive":
    ERR['alpha(ellipse)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR['alpha*m0/m(ellipse)'] = [np.round(0.05 * m0_ellipse_50_c0 / (dimprod_50), 5),
                                  np.round(0.05 * m0_ellipse_50_c1 / (dimprod_50), 5),
                                  np.round(0.05 * m0_ellipse_50_c2 / (dimprod_50), 5)]

  # ramp
  ERR['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="ramp", method=method, shape_spec=spec_ramp_50,
                  alpha=alpha, tail=tail))
    ERR['ramp'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="ramp", method=method, shape_spec=spec_ramp_50,
                  alpha=alpha, tail=tail))
    ERR['ramp'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="ramp", method=method, shape_spec=spec_ramp_50,
                  alpha=alpha, tail=tail))

  # ramp smth
  ERR['ramp(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp(smth)'][0].append(
      error_check(mode=mode, dim=dim_50, threshold=c[0], shape="ramp", method=method, shape_spec=spec_ramp_50_smth,
                  alpha=alpha, tail=tail))
    ERR['ramp(smth)'][1].append(
      error_check(mode=mode, dim=dim_50, threshold=c[1], shape="ramp", method=method, shape_spec=spec_ramp_50_smth,
                  alpha=alpha, tail=tail))
    ERR['ramp(smth)'][2].append(
      error_check(mode=mode, dim=dim_50, threshold=c[2], shape="ramp", method=method, shape_spec=spec_ramp_50_smth,
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

  # circle
  ERR2['circle'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="circular", method=method, shape_spec=spec_cir_100,
                  alpha=alpha,
                  tail=tail))
    ERR2['circle'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="circular", method=method, shape_spec=spec_cir_100,
                  alpha=alpha,
                  tail=tail))
    ERR2['circle'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="circular", method=method, shape_spec=spec_cir_100,
                  alpha=alpha,
                  tail=tail))

  # circle smth
  ERR2['circle(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle(smth)'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="circular", method=method, shape_spec=spec_cir_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['circle(smth)'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="circular", method=method, shape_spec=spec_cir_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['circle(smth)'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="circular", method=method, shape_spec=spec_cir_100_smth,
                  alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['alpha(circle)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(circle)'] = [np.round(0.05 * m0_circular_100_c0 / (dimprod_100), 5),
                                  np.round(0.05 * m0_circular_100_c1 / (dimprod_100), 5),
                                  np.round(0.05 * m0_circular_100_c2 / (dimprod_100), 5)]

  # ellipse
  ERR2['ellipse'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ellipse'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="ellipse", method=method, shape_spec=spec_elp_100,
                  alpha=alpha,
                  tail=tail))
    ERR2['ellipse'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="ellipse", method=method, shape_spec=spec_elp_100,
                  alpha=alpha,
                  tail=tail))
    ERR2['ellipse'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="ellipse", method=method, shape_spec=spec_elp_100,
                  alpha=alpha,
                  tail=tail))

  # ellipse smth
  ERR2['ellipse(smth)'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR2['ellipse(smth)'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="ellipse", method=method, shape_spec=spec_elp_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['ellipse(smth)'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="ellipse", method=method, shape_spec=spec_elp_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['ellipse(smth)'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="ellipse", method=method, shape_spec=spec_elp_100_smth,
                  alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['alpha(ellipse)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(ellipse)'] = [np.round(0.05 * m0_ellipse_100_c0 / (dimprod_100), 5),
                                   np.round(0.05 * m0_ellipse_100_c1 / (dimprod_100), 5),
                                   np.round(0.05 * m0_ellipse_100_c2 / (dimprod_100), 5)]

  # ramp 100*100
  ERR2['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="ramp", method=method, shape_spec=spec_ramp_100,
                  alpha=alpha, tail=tail))
    ERR2['ramp'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="ramp", method=method, shape_spec=spec_ramp_100,
                  alpha=alpha, tail=tail))
    ERR2['ramp'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="ramp", method=method, shape_spec=spec_ramp_100,
                  alpha=alpha, tail=tail))
  # ramp_smth 50*50
  ERR2['ramp(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp(smth)'][0].append(
      error_check(mode=mode, dim=dim_100, threshold=c[0], shape="ramp", method=method, shape_spec=spec_ramp_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['ramp(smth)'][1].append(
      error_check(mode=mode, dim=dim_100, threshold=c[1], shape="ramp", method=method, shape_spec=spec_ramp_100_smth,
                  alpha=alpha, tail=tail))
    ERR2['ramp(smth)'][2].append(
      error_check(mode=mode, dim=dim_100, threshold=c[2], shape="ramp", method=method, shape_spec=spec_ramp_100_smth,
                  alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['alpha(ramp)'] = [0.05, 0.05, 0.05]
  if method == "BH":
    ERR2['alpha*m0/m(ramp)'] = [np.round(0.05 * m0_ramp_100_c0 / (100 * 100), 5),
                                np.round(0.05 * m0_ramp_100_c1 / (100 * 100), 5),
                                np.round(0.05 * m0_ramp_100_c2 / (100 * 100), 5)]

  ERR2_key_calc = [list(ERR2.keys())[i] for i in [1, 2, 4, 5, 7, 8]]
  ERR2.update({n: np.round(np.nanmean(ERR2[n], axis=1), 4) for n in ERR2_key_calc})

  return (ERR, ERR2)




### random field generator
def gen_2D(dim, shape, shape_spec, truncate=3):
  fwhm_noise = shape_spec['fwhm_noise']
  std = shape_spec['std']
  nsubj = dim[0]
  mu = np.zeros(dim)

  # signal
  if shape == "ramp":
    mu = ramp_2D(dim=dim, shape_spec=shape_spec)
  else:
    mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)

  # noise
  noise = np.random.randn(*dim) * std
  sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma_noise, truncate=truncate)  #smoothing

  data = np.array(mu + noise, dtype='float')
  return(data, mu)

def ramp_2D(dim, shape_spec):
  nsubj = dim[0]
  direction = shape_spec['direction']
  mag = shape_spec['mag']
  std = shape_spec['std']

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  if direction == 1: #horizontal
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


def conf_plot_agg(threshold, method, r=0.5, std = 5, mag = 3, tail="two", _min=0, _max=3, fontsize = 25, figsize=(30, 20)):
  cmap1 = colors.ListedColormap(['black', 'blue'])
  cmap2 = colors.ListedColormap(['none', 'yellow'])
  cmap3 = colors.ListedColormap(['none', 'red'])
  dim_100 = (80,100,100)
  f50 = 10
  f100 = 10*2
  spec_cir_100 = {'a':r, 'b':r, 'std':std,'mag':mag, 'fwhm_noise':0, 'fwhm_signal':f100}
  spec_elp_100 = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':0, 'fwhm_signal':f100}
  spec_ramp_100 = {'direction':1, 'std':std,'mag':(0,mag), 'fwhm_noise':0}
  spec_cir_100_smth = {'a':r, 'b':r, 'std':std, 'mag':mag, 'fwhm_noise':3, 'fwhm_signal':f100}
  spec_elp_100_smth = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':3, 'fwhm_signal':f100}
  spec_ramp_100_smth = {'direction':1, 'std':std, 'mag':(0,mag), 'fwhm_noise':3}

  circular_100, mu_circular_100 = gen_2D(dim_100, shape="ellipse", shape_spec=spec_cir_100)
  ellipse_100, mu_ellipse_100 = gen_2D(dim_100, shape="ellipse", shape_spec=spec_elp_100)
  ramp_100, mu_ramp_100 = gen_2D(dim_100, shape="ramp", shape_spec=spec_ramp_100)
  circular_100_smth, _ = gen_2D(dim_100, shape="ellipse", shape_spec=spec_cir_100_smth)
  ellipse_100_smth, _ = gen_2D(dim_100, shape="ellipse", shape_spec=spec_elp_100_smth)
  ramp_100_smth, _ = gen_2D(dim_100, shape="ramp", shape_spec=spec_ramp_100_smth)


  cmap = colors.ListedColormap(['black', 'blue', 'yellow', 'red'])
  fig, axs = plt.subplots(2,3, figsize=figsize)

  im = axs[0, 0].imshow(fdr_cope(data=circular_100, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[0, 0].imshow(fdr_cope(data=circular_100, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[0, 0].imshow(fdr_cope(data=circular_100, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[0, 0].set_title("circle", fontsize = fontsize)

  im = axs[0, 1].imshow(fdr_cope(data=ellipse_100, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[0, 1].imshow(fdr_cope(data=ellipse_100, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[0, 1].imshow(fdr_cope(data=ellipse_100, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[0, 1].set_title("ellipse", fontsize = fontsize)

  im = axs[0, 2].imshow(fdr_cope(data=ramp_100, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[0, 2].imshow(fdr_cope(data=ramp_100, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[0, 2].imshow(fdr_cope(data=ramp_100, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[0, 2].set_title("ramp", fontsize = fontsize)

  im = axs[1, 0].imshow(fdr_cope(data=circular_100_smth, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[1, 0].imshow(fdr_cope(data=circular_100_smth, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[1, 0].imshow(fdr_cope(data=circular_100_smth, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[1, 0].set_title("circle(smoothed)", fontsize = fontsize)


  im = axs[1, 1].imshow(fdr_cope(data=ellipse_100_smth, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[1, 1].imshow(fdr_cope(data=ellipse_100_smth, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[1, 1].imshow(fdr_cope(data=ellipse_100_smth, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[1, 1].set_title("ellipse(smoothed)", fontsize = fontsize)

  im = axs[1, 2].imshow(fdr_cope(data=ramp_100_smth, method=method, threshold=threshold, tail=tail)[0], cmap=cmap1)
  im = axs[1, 2].imshow(fdr_cope(data=ramp_100_smth, method=method, threshold=threshold, tail=tail)[2], cmap=cmap2)
  im = axs[1, 2].imshow(fdr_cope(data=ramp_100_smth, method=method, threshold=threshold, tail=tail)[1], cmap=cmap3)
  axs[1, 2].set_title("ramp(smoothed)", fontsize = fontsize)

  plt.show()