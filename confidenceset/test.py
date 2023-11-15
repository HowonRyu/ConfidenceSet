import numpy as np
import sys
import scipy
from .confset import *
from .random_field_generator import *
### FDR and power tests


def error_check(mode, dim, threshold, method, shape, std=None, shape_spec=None, alpha=0.05):
  """
  checks FDR and FNDR with simulation

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
    "joint" or "separate"
  shape : str
    "ramp" or "ellipse"
  shape_spec : dict
    dictionary containing shape specs
  alpha : int
    [0, 1] alpha level


  Returns
  -------
  ERR : list or int
    corresponding error rates

  Examples
  --------
  ERR = error_check(mode="FNDR", dim=(80,50,50),threshold=4,
  method="joint", shape="circular", shape_spec=spec_cir_50_smth, alpha=0.05)

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  if shape == 'noise':
    data = np.random.randn(*dim) * std
    mu = np.zeros((dim[1], dim[2]))

  else:
    data, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)

  Ac = mu > threshold
  AcC = 1 - Ac
  Acbar = mu >= threshold
  AcbarC = 1-Acbar
  m = dim[1] * dim[2]

  if method == "separate":
    lower, upper, Achat, all_sets, n_rej = fdr_confset(data=data, threshold=threshold, method="separate", alpha=alpha,
             k=2, alpha0=alpha / 4, alpha1=alpha / 2)
  elif method == "joint":
    lower, upper, Achat, all_sets, n_rej = fdr_confset(data=data, threshold=threshold, method="joint", alpha=alpha*2,
             k=2, alpha0=(alpha*2)/4, alpha1=(alpha*2)/2)


  if mode == "FDR":
    if method == "separate":
      ERR = [None, None]

      if n_rej[1] == 0:
        ERR[1] = 0
        #print("no rejection (upper)")
      else:
        upper_nom = np.sum(np.maximum(upper - Ac.astype(int), 0))
        upper_denom = np.sum(upper)
        upper_ERR = upper_nom / upper_denom
        ERR[1] = upper_ERR
        #print(f'upper rej={n_rej[1]}, upper_denom={upper_denom}')

      if n_rej[0] == 0:
        ERR[0] = 0
        #print("no rejection (lower)")
      else:
        lower_nom = np.sum(np.maximum(Acbar.astype(int) - lower, 0))
        lower_denom = np.sum(1-lower)
        lower_ERR = lower_nom / lower_denom
        ERR[0] = lower_ERR
        #print(f'lower rej={n_rej[0]}, lower_denom={lower_denom}')


      return ERR

    elif method == "joint":
      if n_rej == 0:
        ERR = 0
        #print("no rejection")

      else:
        nom1 = np.sum(np.maximum(upper-Ac.astype(int),0))
        nom2 = np.sum(np.maximum(Acbar.astype(int)-lower,0))
        nom = nom1 + nom2

        denom = np.sum(upper) + np.sum(1-lower)
        #print(f'joint: denom={denom}, n_rej={n_rej}')

        ERR = nom / denom
      return ERR

  elif mode == "FNDR":
    if method == "separate":
      ERR = [None, None]

      if n_rej[1] == m:
        ERR[1] = 0
        #print("all rejection (upper)")
      else:
        upper_nom = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        upper_denom = np.sum(1-upper)
        upper_ERR = upper_nom / upper_denom
        ERR[1] = upper_ERR
        #print(f'upper non rej={m-n_rej[1]}, upper_denom={upper_denom}')

      if n_rej[0] == m:
        ERR[0] = 0
        #print("all rejection (lower)")
      else:
        lower_nom = np.sum(np.maximum(lower - Acbar.astype(int), 0))
        lower_denom = np.sum(lower)
        lower_ERR = lower_nom / lower_denom
        ERR[0] = lower_ERR
        #print(f'lower non rej={m-n_rej[0]}, lower_denom={lower_denom}')

      return ERR

    elif method == "joint":
      if n_rej == 2*m:
        ERR = 0
        #print("all rejection")
      else:
        nom1 = np.sum(np.maximum(Ac.astype(int) - upper,0))
        nom2 = np.sum(np.maximum(lower - Acbar.astype(int),0))
        nom = nom1 + nom2

        denom = np.sum(1-upper) + np.sum(lower)
        #print(f'joint non rej={2*m-n_rej}, joint_denom: denom={denom}')

        ERR = nom / denom
      return ERR

  def error_check_sim_table(sim_num, mode, method, shape, shape_spec, c, dim, c_marg=0.2, alpha=0.05):
    """
    produces table for FDR, and FNDR simulation result

    Parameters
    ----------
    sim_num : int
      simulation number
    mode : str
      options for error rate "FDR" or "FNDR"
    method : str
      "separate" or "joint"
    shape : str
      "ramp" or "ellipse"
    shape_spec : dict
      dictionary containing shape specs
    c : list
      list of thresholds
    dim : int
      dimension of the image (N, W, H)
    c_marg : int
      margin allowed for the threshold
    alpha : int
      [0, 1] alpha level

    Returns
    -------
    sim_table : array
      simulated error rate result

    Examples
    --------
    error_check_sim_table(sim_num=sim_num, mode=mode, method="joint",
                                      shape=shape, shape_spec=shape_spec, c=c,
                                      dim=dim, c_marg=0.2, alpha=0.05)

    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """

    if method == "separate":
      sim_table_lower = np.empty([len(c), sim_num])
      sim_table_upper = np.empty([len(c), sim_num])

      for jidx, j in enumerate(c):
        sim_temp_upper = list()
        sim_temp_lower = list()
        for i in np.arange(sim_num):
          sim_temp_upper.append(error_check(mode=mode, dim=dim, threshold=j, shape=shape,
                                            method=method, shape_spec=shape_spec, alpha=alpha)[1])
          sim_temp_lower.append(error_check(mode=mode, dim=dim, threshold=j, shape=shape,
                                            method=method, shape_spec=shape_spec, alpha=alpha)[0])
        sim_table_lower[jidx, :] = sim_temp_lower
        sim_table_upper[jidx, :] = sim_temp_upper
      return sim_table_lower, sim_table_upper

    if method == "joint":
      sim_table = np.empty([len(c), sim_num])
      for jidx, j in enumerate(c):
        sim_temp = list()
        for i in np.arange(sim_num):
          sim_temp.append(error_check(mode=mode, dim=dim, threshold=j, shape=shape,
                                      method=method, shape_spec=shape_spec, alpha=alpha))
        sim_table[jidx, :] = sim_temp
      return sim_table



