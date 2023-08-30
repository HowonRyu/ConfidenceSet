import numpy as np
import scipy
from .random_field_generator import *

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

    return (rejection_ind, rejection_locs, nrejections)


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
    sq_xbar = np.mean(data ** 2, axis=0)

    est_var = (n_subj / (n_subj - 1)) * (sq_xbar - (xbar ** 2))
    std_dev = np.sqrt(est_var)

    tstat = (np.sqrt(n_subj) * xbar) / std_dev
    cohensd = xbar / std_dev

    return (tstat)


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

    return (rejection_ind, rejection_locs, nrejections)


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
    return (y)


def fdr_cope_temp1(data, threshold, method, alpha=0.05,
             k=2, tail = "two", alpha0=0.05/4, alpha1=0.05/2):
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
  spec_cir_100_smth = {'a':0.5, 'b':0.5, 'std':7, 'mag':3, 'fwhm_noise':3, 'fwhm_signal':10}
  outer_set, inner_set, Achat, plot_add, n_rej = fdr_cope_temp1(data=circular_100_smth, method=method, tail="one", alpha=0.05, threshold=3)

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

  if tail == "one":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    #outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    rejection_ind = np.full(np.prod(inner_pvals.shape), 0)
    if method == "adaptive":
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
      #outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    elif method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha)
      #outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha)
    n_rej = inner_n_rej
    outer_set = None
    inner_set = Achat * inner_rejection_ind
    plot_add =  inner_set + Achat

  elif tail == "two":
    pvals_upper = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    pvals_lower = 1 - pvals_upper
    pvals = np.concatenate((pvals_upper, pvals_lower))

    rejection_ind = np.full(len(pvals.shape), 0)
    if method == "adaptive":
      rejection_ind, _, n_rej = fdr_adaptive(pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    elif method == "BH":
      rejection_ind, _, n_rej = fdr_BH(pvals, alpha)

    rejection_ind_upper, rejection_ind_lower = np.array_split(rejection_ind, 2)

    outer_set = 1 - Achat_C * rejection_ind_lower
    inner_set = Achat * rejection_ind_upper
    plot_add = outer_set + inner_set + Achat


  return(outer_set, inner_set, Achat, plot_add, n_rej)



def fdr_cope_temp2(data, threshold, method, alpha=0.05,
             k=2, tail="two", alpha0=0.05/4, alpha1=0.05/2):
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
  spec_cir_100_smth = {'a':0.5, 'b':0.5, 'std':7, 'mag':3, 'fwhm_noise':3, 'fwhm_signal':10}
  outer_set, inner_set, Achat, plot_add, n_rej = fdr_cope_temp1(data=circular_100_smth, method=method, tail="one", alpha=0.05, threshold=3)

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

  if tail == "one":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    #outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    rejection_ind = np.full(np.prod(inner_pvals.shape), 0)
    if method == "adaptive":
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
      #outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    elif method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha)
      #outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha)
    n_rej = [inner_n_rej]
    outer_set = None
    inner_set = Achat * inner_rejection_ind
    plot_add =  inner_set + Achat

  elif tail == "two":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    rejection_ind = np.full(np.prod(inner_pvals.shape), 0)
    if method == "adaptive":
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0/2, alpha1=alpha1/2)
      outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0/2, alpha1=alpha1/2)
    elif method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha/2)
      outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha/2)
    n_rej = [inner_n_rej, outer_n_rej]
    outer_set = 1 - Achat_C * outer_rejection_ind
    inner_set = Achat * inner_rejection_ind
    plot_add = outer_set + inner_set + Achat

  return(outer_set, inner_set, Achat, plot_add, n_rej)

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
  lower, upper = fdr_cope(data, threshold=2, method="BH", alpha=0.05, tail="two")
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

  if tail == "two":
    pvals = 2 * (1 - scipy.stats.t.cdf(abs(data_tstat), df=nsubj - 1))
    rejection_ind = np.full(np.prod(pvals.shape), 0)
    if method == "adaptive":
      rejection_ind, _, n_rej = fdr_adaptive(pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    if method == "BH":
      rejection_ind, _, n_rej = fdr_BH(pvals, alpha)
    outer_set = 1 - Achat_C * rejection_ind
    inner_set = Achat * rejection_ind
    plot_add = outer_set + inner_set + Achat
    return (outer_set, inner_set, Achat, plot_add, n_rej)

  if tail == "one":
    inner_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    #outer_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    rejection_ind = np.full(np.prod(inner_pvals.shape), 0)
    if method == "adaptive":
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
      #outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    if method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha)
      #outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha)
    inner_set = Achat * inner_rejection_ind
    outer_set = None
    plot_add = inner_set + Achat
    return (outer_set, inner_set, Achat, plot_add, inner_n_rej)
