import numpy as np
import sys
import scipy
from .confset import *
from .random_field_generator import *
from .plot import *
### FDR and power tests


def error_check_temp(temp, mode, dim, threshold, method, shape, shape_spec=None, alpha=0.05, tail="two"):
  """
  checks FDR, FNDR with simulation

  Parameters
  ----------
  temp : str
    options for creating confidence set "0", "1" or "2"
  mode : str
    options for error rate "FDR" or "FNDR"
  dim : int
    dimension of the image (N, W, H)
  threshold : int
    threshold to be used for sub-setting
  method : str
    "BH" or "Adaptive"
  shape : str
    "ramp" or "ellipse"
  shape_spec : dict
    dictionary containing shape specs
  alpha : int
    [0, 1] alpha level
  tail : str
    "one" or "two"

  Returns
  -------
  ERR : int
    corresponding error rate

  Examples
  --------
  testERR, testlower, testupper, testAc = error_check_temp(temp="1", mode="fndr", dim=(80,50,50),
                                                         threshold=4, method="BH", shape="circular", shape_spec=spec_cir_50_smth, alpha=0.05, tail="two")

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  if shape == 'noise':
    data = np.random.randn(*dim) * std
    mu = np.zeros((dim[1], dim[2]))

  else:
    data, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)

  Ac = mu >= threshold
  AcC = 1 - Ac

  if temp == "0":
    lower, upper, Achat, all_sets, n_rej = fdr_cope(data, method=method, threshold=threshold, alpha=alpha, tail=tail)
  elif temp == "1":
    lower, upper, Achat, all_sets, n_rej = fdr_cope_temp1(data, method=method, threshold=threshold, alpha=alpha,
                                                          tail=tail)
  elif temp == "2":
    lower, upper, Achat, all_sets, n_rej = fdr_cope_temp2(data, method=method, threshold=threshold, alpha=alpha / 2,
                                                          tail=tail)

  if n_rej == 0:
    ERR = 0
    # return(ERR, lower, upper, Ac)
    return (ERR)

  if temp == "0" or temp == "1":
    ERR = -1

    if mode == "fdr":
      if tail == "one":
        numer = np.sum(np.maximum(upper - Ac.astype(int), 0))
        denom = np.sum(upper)

      elif tail == "two":
        numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
        denom = np.sum(np.minimum(upper + (1 - lower), 1))


    elif mode == "fndr":
      if tail == "one":
        numer = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        denom = np.sum(1 - upper)

      elif tail == "two":
        numer = np.sum(np.minimum(np.maximum(Ac.astype(int) - upper, 0) + np.maximum(lower - Ac.astype(int), 0), 1))
        # denom = np.sum(np.minimum( (1-upper) + lower, 1))
        denom = dim[1] * dim[2]

    if denom == 0:
      ERR = 0
    else:
      ERR = numer / denom

    # return(ERR, lower, upper, Ac)
    return (ERR)


  elif temp == "2":
    ERR1 = -1
    ERR2 = -1
    if mode == "fdr":
      if tail == "one":
        numer1 = np.sum(np.maximum(upper - Ac.astype(int), 0))
        denom1 = np.sum(upper)
        numer2 = 0
        denom2 = 1

      elif tail == "two":
        numer1 = np.sum(np.maximum(upper - Ac.astype(int), 0))
        denom1 = np.sum(upper)
        numer2 = np.sum(np.maximum(Ac.astype(int) - lower, 0))
        denom2 = np.sum(1 - lower)


    elif mode == "fndr":
      if tail == "one":
        numer1 = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        denom1 = np.sum(1 - upper)
        numer2 = 0
        denom2 = 1

      elif tail == "two":
        numer1 = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        numer2 = np.sum(np.maximum(lower - Ac.astype(int), 0))
        denom1 = np.sum(1 - upper)
        denom2 = np.sum(lower)

    if denom1 == 0:
      ERR1 = 0
    else:
      ERR1 = numer1 / denom1

    if denom2 == 0:
      ERR2 = 0
    else:
      ERR2 = numer2 / denom2

    # return(ERR1+ERR2, lower, upper, Ac)
    return (ERR1 + ERR2)

  def error_check_sim_table(sim_num, temp, mode, method, shape, shape_spec, c, dim, c_marg=0.2, tail="two", alpha=0.05,
                            alpha0=0.05 / 4, alpha1=0.05 / 2):
      sim_table = np.empty([len(c), sim_num])
      for jidx, j in enumerate(c):
          sim_temp = list()
          for i in np.arange(sim_num):
              sim_temp.append(error_check_temp(temp=temp, mode=mode, dim=dim, threshold=j, shape=shape, method=method,
                                               shape_spec=shape_spec,
                                               alpha=alpha, tail=tail))
          sim_table[jidx, :] = sim_temp
      return sim_table



