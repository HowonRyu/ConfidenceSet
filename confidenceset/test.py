import numpy as np
import sys
import scipy
from .confset import *
from .random_field_generator import *
### FDR and power tests


def error_check(data, mu, mode, threshold, method, alpha=0.05):
  """
  returns the error rate (FDR or FNDR) for the input data

  Parameters
  ----------
  data : array
    2D field input of dim (N, W, H)
  mu : array
    2D signal field input for ground truth of dim (N, W, H)
  mode : str
    options for error rate "FDR" or "FNDR"
  threshold : float
    c, the threshold to be used for inference
  method : str
    "joint", "separate_adaptive" or "separate_BH"
  alpha : float
    [0, 1] alpha level

  Returns
  -------
  ERR : list or float
    corresponding error rates

  Examples
  --------
  field, signal = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
  ERR = error_check(mode="FNDR", data=field, mu=signal, threshold=4, method="joint", alpha=0.05)

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """

  Ac = mu > threshold
  AcC = 1 - Ac
  Acbar = mu >= threshold
  AcbarC = 1-Acbar
  dim = data.shape
  m = dim[1] * dim[2]

  if method == "separate_adaptive":
    lower, upper, Achat, all_sets, n_rej = fdr_confset(data=data, threshold=threshold, method="separate_adaptive", alpha=alpha,
             k=2, alpha0=alpha / 4, alpha1=alpha / 2)
  if method == "separate_BH":
    lower, upper, Achat, all_sets, n_rej = fdr_confset(data=data, threshold=threshold, method="separate_BH", alpha=alpha)
  if method == "joint":
    lower, upper, Achat, all_sets, n_rej = fdr_confset(data=data, threshold=threshold, method="joint", alpha=alpha*2,
             k=2, alpha0=(alpha*2)/4, alpha1=(alpha*2)/2)


  if mode == "FDR":
    if method == "separate_adaptive" or method == "separate_BH":
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

    if method == "joint":
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

  if mode == "FNDR":
    if method == "separate_adaptive" or method == "separate_BH":
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

    if method == "joint":
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





