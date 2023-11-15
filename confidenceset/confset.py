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
      shows whether the voxel is rejected
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
    rejection_ind, rejection_locs, nrejections = fdr_BH(pvals, alpha=0.05)

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
    #cohensd = xbar / std_dev

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
      shows whether the voxel is rejected
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


def fdr_confset(data, threshold, method="separate", alpha=0.05,
             k=2, alpha0=0.05 / 4, alpha1=0.05 / 2):
    """
    sub-setting the confidence set controlling for FDR

    Parameters
    ----------
    data : int
      array of voxels
    method : str
      either "separate" or "joint"
    threshold : int
      threshold to be used for sub-setting
    alpha : int
      alpha level
    k : int
      kappa level for the adaptive procedure
    alpha0 : int
      alpha0 level for the adaptive procedure
    alpha1 : int
      alpha1 level for the adaptive procedure


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
    n_rej: list
      number of voxels rejected by the procedure for lower and upper sets or both

    Example
    -------
    nsub = 50
    data = numpy.random.randn(nsub, 100, 100) + 2
    lower, upper, _, _, _ = fdr_cope(data, threshold=2, method="separate", alpha=0.05, tail="two")
    plt.imshow(lower)
    plt.imshow(upper)

    :Authors:
      Samuel Davenport <sdavenport@health.ucsd.edu>
      Howon Ryu <howonryu@ucsd.edu>
    """

    data_tstat = mvtstat(data - threshold)
    data_dim = data.shape
    nsubj = data_dim[0]
    Achat = data_tstat > 0
    # Achat_C = data_tstat <= 0
    # Acbarhat = data_tstat >= 0
    # Acbarhat_C = data_tstat < 0
    upper_pvals = 1 - scipy.stats.t.cdf(data_tstat, df=nsubj - 1)
    lower_pvals = scipy.stats.t.cdf(data_tstat, df=nsubj - 1)

    if method == "separate":
        # upper set
        upper_rej_ind, _, upper_n_rej = fdr_BH(upper_pvals, alpha=alpha)
        upper_set = upper_rej_ind
        # upper_set = Achat * upper_rejection_ind

        # lower set
        lower_rej_ind, _, lower_n_rej = fdr_adaptive(lower_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
        lower_set = 1 - lower_rej_ind
        # lower_set = 1 - (Acbarhat_C * lower_rejection_ind)

        n_rej = [lower_n_rej, upper_n_rej]

        plot_add = upper_set + lower_set + Achat
        return lower_set, upper_set, Achat, plot_add, n_rej

    if method == "joint":
        pvals = np.concatenate((upper_pvals, lower_pvals))
        rejection_ind = np.full(np.prod(pvals.shape), 0)
        rejection_ind, _, n_rej = fdr_BH(pvals, alpha)
        upper_rej_ind, lower_rej_ind = np.array_split(rejection_ind, 2)

        # upper set
        upper_set = upper_rej_ind

        # lower set
        lower_set = 1 - lower_rej_ind

        plot_add = upper_set + lower_set + Achat

        return lower_set, upper_set, Achat, plot_add, n_rej
    else:
        return("wrong method")