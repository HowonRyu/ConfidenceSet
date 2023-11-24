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

def error_check_sim_temp(temp, sim_num, mode, method, c, c_marg=0.2, std=5, tail="two", alpha=0.05, alpha0=0.05 / 4,
                         alpha1=0.05 / 2):
  dim_100 = (80, 100, 100)
  dimprod_100 = dim_100[1] * dim_100[2]
  dim_50 = (80, 50, 50)
  dimprod_50 = dim_50[1] * dim_50[2]
  up0, lo0 = c[0] + c_marg, c[0] - c_marg
  up1, lo1 = c[1] + c_marg, c[1] - c_marg
  up2, lo2 = c[2] + c_marg, c[2] - c_marg
  r = 0.5
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
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method,
                       shape_spec=spec_cir_50,
                       alpha=alpha,
                       tail=tail))
    ERR['circle'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method,
                       shape_spec=spec_cir_50,
                       alpha=alpha,
                       tail=tail))
    ERR['circle'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method,
                       shape_spec=spec_cir_50,
                       alpha=alpha,
                       tail=tail))

  # circle smth
  ERR['circle(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['circle(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method,
                       shape_spec=spec_cir_50_smth,
                       alpha=alpha,
                       tail=tail))
    ERR['circle(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method,
                       shape_spec=spec_cir_50_smth,
                       alpha=alpha,
                       tail=tail))
    ERR['circle(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method,
                       shape_spec=spec_cir_50_smth,
                       alpha=alpha,
                       tail=tail))

  if method == "adaptive":
    ERR['circle-expected'] = [alpha, alpha, alpha]

  if method == "BH":
    if temp == "1":
      ERR['circle-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      if tail == "two":
        ERR['circle-expected'] = [np.round(((alpha / 2) * np.sum(mu_circular_50 > c[0]) / dimprod_50) + (
                  (alpha / 2) * np.sum(mu_circular_50 < c[0]) / dimprod_50), 5),
                                  np.round(((alpha / 2) * np.sum(mu_circular_50 > c[1]) / dimprod_50) + (
                                            (alpha / 2) * np.sum(mu_circular_50 < c[1]) / dimprod_50), 5),
                                  np.round(((alpha / 2) * np.sum(mu_circular_50 > c[2]) / dimprod_50) + (
                                            (alpha / 2) * np.sum(mu_circular_50 < c[2]) / dimprod_50), 5)]
      elif tail == "one":
        ERR['circle-expected'] = [np.round(((alpha / 2) * np.sum(mu_circular_50 > c[0]) / dimprod_50), 5),
                                  np.round(((alpha / 2) * np.sum(mu_circular_50 > c[1]) / dimprod_50), 5),
                                  np.round(((alpha / 2) * np.sum(mu_circular_50 > c[2]) / dimprod_50), 5)]
    else:
      ERR['circle-expected'] = [np.round(alpha * m0_circular_50_c0 / (dimprod_50), 5),
                                np.round(alpha * m0_circular_50_c1 / (dimprod_50), 5),
                                np.round(alpha * m0_circular_50_c2 / (dimprod_50), 5)]

  # ellipse
  ERR['ellipse'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ellipse'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="ellipse", method=method,
                       shape_spec=spec_elp_50,
                       alpha=alpha,
                       tail=tail))
    ERR['ellipse'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="ellipse", method=method,
                       shape_spec=spec_elp_50,
                       alpha=alpha,
                       tail=tail))
    ERR['ellipse'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="ellipse", method=method,
                       shape_spec=spec_elp_50,
                       alpha=alpha,
                       tail=tail))

  # ellipse smth
  ERR['ellipse(smth)'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR['ellipse(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="circular", method=method,
                       shape_spec=spec_elp_50_smth,
                       alpha=alpha,
                       tail=tail))
    ERR['ellipse(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="circular", method=method,
                       shape_spec=spec_elp_50_smth,
                       alpha=alpha,
                       tail=tail))
    ERR['ellipse(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="circular", method=method,
                       shape_spec=spec_elp_50_smth,
                       alpha=alpha,
                       tail=tail))

  if method == "adaptive":
    ERR['ellipse-expected'] = [alpha, alpha, alpha]
  if method == "BH":
    if temp == "1":
      ERR['ellipse-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      if tail == "two":
        ERR['ellipse-expected'] = [np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[0]) / dimprod_50) + (
                  (alpha / 2) * np.sum(mu_ellipse_50 < c[0]) / dimprod_50), 5),
                                   np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[1]) / dimprod_50) + (
                                             (alpha / 2) * np.sum(mu_ellipse_50 < c[1]) / dimprod_50), 5),
                                   np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[2]) / dimprod_50) + (
                                             (alpha / 2) * np.sum(mu_ellipse_50 < c[2]) / dimprod_50), 5)]
      elif tail == "one":
        ERR['ellipse-expected'] = [np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[0]) / dimprod_50), 5),
                                   np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[1]) / dimprod_50), 5),
                                   np.round(((alpha / 2) * np.sum(mu_ellipse_50 > c[2]) / dimprod_50), 5)]
    else:
      ERR['ellipse-expected'] = [np.round(alpha * m0_ellipse_50_c0 / (dimprod_50), 5),
                                 np.round(alpha * m0_ellipse_50_c1 / (dimprod_50), 5),
                                 np.round(alpha * m0_ellipse_50_c2 / (dimprod_50), 5)]

  # ramp
  ERR['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="ramp", method=method,
                       shape_spec=spec_ramp_50,
                       alpha=alpha, tail=tail))
    ERR['ramp'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="ramp", method=method,
                       shape_spec=spec_ramp_50,
                       alpha=alpha, tail=tail))
    ERR['ramp'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="ramp", method=method,
                       shape_spec=spec_ramp_50,
                       alpha=alpha, tail=tail))

  # ramp smth
  ERR['ramp(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR['ramp(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[0], shape="ramp", method=method,
                       shape_spec=spec_ramp_50_smth,
                       alpha=alpha, tail=tail))
    ERR['ramp(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[1], shape="ramp", method=method,
                       shape_spec=spec_ramp_50_smth,
                       alpha=alpha, tail=tail))
    ERR['ramp(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_50, threshold=c[2], shape="ramp", method=method,
                       shape_spec=spec_ramp_50_smth,
                       alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR['ramp-expected'] = [alpha, alpha, alpha]
  if method == "BH":
    if temp == "1":
      ERR['ramp-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      if tail == "two":
        ERR['ramp-expected'] = [np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[0]) / dimprod_50) + (
                  (alpha / 2) * np.sum(mu_ramp_50 < c[0]) / dimprod_50), 5),
                                np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[1]) / dimprod_50) + (
                                          (alpha / 2) * np.sum(mu_ramp_50 < c[1]) / dimprod_50), 5),
                                np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[2]) / dimprod_50) + (
                                          (alpha / 2) * np.sum(mu_ramp_50 < c[2]) / dimprod_50), 5)]
      elif tail == "one":
        ERR['ramp-expected'] = [np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[0]) / dimprod_50), 5),
                                np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[1]) / dimprod_50), 5),
                                np.round(((alpha / 2) * np.sum(mu_ramp_50 > c[2]) / dimprod_50), 5)]
    else:
      ERR['ramp-expected'] = [np.round(alpha * m0_ramp_50_c0 / (dimprod_50), 5),
                              np.round(alpha * m0_ramp_50_c1 / (dimprod_50), 5),
                              np.round(alpha * m0_ramp_50_c2 / (dimprod_50), 5)]

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
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="circular", method=method,
                       shape_spec=spec_cir_100,
                       alpha=alpha,
                       tail=tail))
    ERR2['circle'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="circular", method=method,
                       shape_spec=spec_cir_100,
                       alpha=alpha,
                       tail=tail))
    ERR2['circle'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="circular", method=method,
                       shape_spec=spec_cir_100,
                       alpha=alpha,
                       tail=tail))

  # circle smth
  ERR2['circle(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['circle(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="circular", method=method,
                       shape_spec=spec_cir_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['circle(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="circular", method=method,
                       shape_spec=spec_cir_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['circle(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="circular", method=method,
                       shape_spec=spec_cir_100_smth,
                       alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['circle-expected'] = [alpha, alpha, alpha]
  if method == "BH":
    if temp == "1":
      ERR2['circle-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      if tail == "two":
        ERR2['circle-expected'] = [np.round(((alpha / 2) * np.sum(mu_circular_100 > c[0]) / dimprod_100) + (
                  (alpha / 2) * np.sum(mu_circular_100 < c[0]) / dimprod_100), 5),
                                   np.round(((alpha / 2) * np.sum(mu_circular_100 > c[1]) / dimprod_100) + (
                                             (alpha / 2) * np.sum(mu_circular_100 < c[1]) / dimprod_100), 5),
                                   np.round(((alpha / 2) * np.sum(mu_circular_100 > c[2]) / dimprod_100) + (
                                             (alpha / 2) * np.sum(mu_circular_100 < c[2]) / dimprod_100), 5)]
      elif tail == "one":
        ERR2['circle-expected'] = [np.round(((alpha / 2) * np.sum(mu_circular_100 > c[0]) / dimprod_100), 5),
                                   np.round(((alpha / 2) * np.sum(mu_circular_100 > c[1]) / dimprod_100), 5),
                                   np.round(((alpha / 2) * np.sum(mu_circular_100 > c[2]) / dimprod_100), 5)]
    else:
      ERR2['circle-expected'] = [np.round(alpha * m0_circular_100_c0 / (dimprod_100), 5),
                                 np.round(alpha * m0_circular_100_c1 / (dimprod_100), 5),
                                 np.round(alpha * m0_circular_100_c2 / (dimprod_100), 5)]

  # ellipse
  ERR2['ellipse'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ellipse'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="ellipse", method=method,
                       shape_spec=spec_elp_100,
                       alpha=alpha,
                       tail=tail))
    ERR2['ellipse'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="ellipse", method=method,
                       shape_spec=spec_elp_100,
                       alpha=alpha,
                       tail=tail))
    ERR2['ellipse'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="ellipse", method=method,
                       shape_spec=spec_elp_100,
                       alpha=alpha,
                       tail=tail))

  # ellipse smth
  ERR2['ellipse(smth)'] = [[], [], []]

  for i in np.arange(sim_num):
    ERR2['ellipse(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="ellipse", method=method,
                       shape_spec=spec_elp_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['ellipse(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="ellipse", method=method,
                       shape_spec=spec_elp_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['ellipse(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="ellipse", method=method,
                       shape_spec=spec_elp_100_smth,
                       alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['ellipse-expected'] = [alpha, alpha, alpha]
  if method == "BH":
    if temp == "1":
      ERR2['ellipse-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      ERR2['ellipse-expected'] = [np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[0]) / dimprod_100) + (
                (alpha / 2) * np.sum(mu_ellipse_100 < c[0]) / dimprod_100), 5),
                                  np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[1]) / dimprod_100) + (
                                            (alpha / 2) * np.sum(mu_ellipse_100 < c[1]) / dimprod_100), 5),
                                  np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[2]) / dimprod_100) + (
                                            (alpha / 2) * np.sum(mu_ellipse_100 < c[2]) / dimprod_100), 5)]
      if tail == "one":
        ERR2['ellipse-expected'] = [np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[0]) / dimprod_100), 5),
                                    np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[1]) / dimprod_100), 5),
                                    np.round(((alpha / 2) * np.sum(mu_ellipse_100 > c[2]) / dimprod_100), 5)]
    else:
      ERR2['ellipse-expected'] = [np.round(0.05 * m0_ellipse_100_c0 / (dimprod_100), 5),
                                  np.round(0.05 * m0_ellipse_100_c1 / (dimprod_100), 5),
                                  np.round(0.05 * m0_ellipse_100_c2 / (dimprod_100), 5)]

  # ramp 100*100
  ERR2['ramp'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="ramp", method=method,
                       shape_spec=spec_ramp_100,
                       alpha=alpha, tail=tail))
    ERR2['ramp'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="ramp", method=method,
                       shape_spec=spec_ramp_100,
                       alpha=alpha, tail=tail))
    ERR2['ramp'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="ramp", method=method,
                       shape_spec=spec_ramp_100,
                       alpha=alpha, tail=tail))
  # ramp_smth 50*50
  ERR2['ramp(smth)'] = [[], [], []]
  for i in np.arange(sim_num):
    ERR2['ramp(smth)'][0].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[0], shape="ramp", method=method,
                       shape_spec=spec_ramp_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['ramp(smth)'][1].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[1], shape="ramp", method=method,
                       shape_spec=spec_ramp_100_smth,
                       alpha=alpha, tail=tail))
    ERR2['ramp(smth)'][2].append(
      error_check_temp(temp=temp, mode=mode, dim=dim_100, threshold=c[2], shape="ramp", method=method,
                       shape_spec=spec_ramp_100_smth,
                       alpha=alpha, tail=tail))

  if method == "adaptive":
    ERR2['ramp-expected'] = [alpha, alpha, alpha]
  if method == "BH":
    if temp == "1":
      ERR2['ramp-expected'] = [0.5 * alpha, 0.5 * alpha, 0.5 * alpha]
    elif temp == "2":
      if tail == "two":
        ERR2['ramp-expected'] = [np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[0]) / dimprod_100) + (
                  (alpha / 2) * np.sum(mu_ramp_100 < c[0]) / dimprod_100), 5),
                                 np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[1]) / dimprod_100) + (
                                           (alpha / 2) * np.sum(mu_ramp_100 < c[1]) / dimprod_100), 5),
                                 np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[2]) / dimprod_100) + (
                                           (alpha / 2) * np.sum(mu_ramp_100 < c[2]) / dimprod_100), 5)]
      elif tail == "one":
        ERR2['ramp-expected'] = [np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[0]) / dimprod_100), 5),
                                 np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[1]) / dimprod_100), 5),
                                 np.round(((alpha / 2) * np.sum(mu_ramp_100 > c[2]) / dimprod_100), 5)]
    else:
      ERR2['ramp-expected'] = [np.round(alpha * m0_ramp_100_c0 / dimprod_100, 5),
                               np.round(alpha * m0_ramp_100_c1 / dimprod_100, 5),
                               np.round(alpha * m0_ramp_100_c2 / dimprod_100, 5)]

  ERR2_key_calc = [list(ERR2.keys())[i] for i in [1, 2, 4, 5, 7, 8]]
  ERR2.update({n: np.round(np.nanmean(ERR2[n], axis=1), 4) for n in ERR2_key_calc})

  return (ERR, ERR2)



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




def table_sim():
  simnum = 500
  cmarg = 0.1
  std=  5
  c = (0.5, 2, 3)
  BH50_fdr, BH100_fdr = error_check_sim_temp(temp="1",mode = "fdr", sim_num=simnum, method="BH", c=c, c_marg=cmarg, std=std, alpha=0.05, alpha0=0.05/4, alpha1=0.05/2)
  AD50_fdr, AD100_fdr = error_check_sim_temp(temp="1", mode = "fdr", sim_num=simnum, method="adaptive", c=c, c_marg=cmarg, std=std, alpha=0.05, alpha0=0.05/4, alpha1=0.05/2)

  print("------------------------------------------------------------------------- BH FDR ----------------------------------------------------------------------------")
  print("----------------------------------------------------------------------imagesize:(50*50)----------------------------------------------------------------------")
  print(tabulate(BH50_fdr, headers='keys'))
  print()
  print("----------------------------------------------------------------------imagesize:(100*100)--------------------------------------------------------------------")
  print(tabulate(BH100_fdr, headers='keys'))

  print("----------------------------------------------------------------------- Adaptive FDR ------------------------------------------------------------------------")
  print("----------------------------------------------------------------------imagesize:(50*50)----------------------------------------------------------------------")
  print(tabulate(AD50_fdr, headers='keys'))
  print()
  print("----------------------------------------------------------------------imagesize:(100*100)--------------------------------------------------------------------")
  print(tabulate(AD100_fdr, headers='keys'))



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
      inner_rejection_ind, _, inner_n_rej = fdr_adaptive(inner_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
      outer_rejection_ind, _, outer_n_rej = fdr_adaptive(outer_pvals, k=k, alpha0=alpha0, alpha1=alpha1)
    elif method == "BH":
      inner_rejection_ind, _, inner_n_rej = fdr_BH(inner_pvals, alpha=alpha)
      outer_rejection_ind, _, outer_n_rej = fdr_BH(outer_pvals, alpha=alpha)
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


def conf_plot_agg_temp(threshold, temp, method, seed=None, r=0.5, std = 5, fwhm_noise=3, fwhm_signal=20, mag = 3, fontsize = 25, figsize=(30, 20), alpha=0.05):

  """
  plots FDR controlling confidence sets for six different random fields

  Parameters
  ----------
  threshold : int
      threshold c
  temp : str
      options for creating confidence set "0", "1" or "2"
  method : str
      "BH" or "adaptive"
  r : int
      radii of ellipses
  std : int
      standard deviation for the noise field N(0, std^2)
  mag : int
      magnitude of the signal
  fontsize : int
  f   ont size for figure
  figsize : tuple
      figure size
  alpha : int
  [0, 1] alpha level

  Examples
  --------
  conf_plot_agg_temp(threshold=2, temp = "1", std=7, method="BH", _min=0, _max=3, fontsize=10, alpha=0.05, figsize = (5,3))

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  if temp == "1":
    fdr_cope_function = fdr_cope_temp1
  elif temp == "2":
    fdr_cope_function = fdr_cope_temp2
  elif temp == "0":
    fdr_cope_function = fdr_cope

  cmap1 = colors.ListedColormap(['black', 'blue'])
  cmap2 = colors.ListedColormap(['none', 'yellow'])
  cmap22 = colors.ListedColormap(['black', 'yellow'])
  cmap3 = colors.ListedColormap(['none', 'red'])
  dim_100 = (80,100,100)


  spec_cir_100_smth = {'a':r, 'b':r, 'std':std, 'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_elp_100_smth = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_ramp_100_smth = {'direction':1, 'std':std, 'mag':(0,mag), 'fwhm_noise':fwhm_noise}

  circular_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_cir_100_smth)
  ellipse_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_elp_100_smth)
  ramp_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ramp", shape_spec=spec_ramp_100_smth)


  cmap = colors.ListedColormap(['black', 'blue', 'yellow', 'red'])
  fig, axs = plt.subplots(1,3, figsize=figsize)

  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two",  alpha=alpha,  threshold=threshold)[0], cmap=cmap1)
  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[2], cmap=cmap2)
  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[1], cmap=cmap3)
  axs[0].set_title("circle", fontsize = fontsize)

  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[2], cmap=cmap2)
  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[1], cmap=cmap3)
  axs[1].set_title("ellipse", fontsize = fontsize)

  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, tail="two", method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  axs[2].set_title("ramp", fontsize = fontsize)

  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha,  threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 0].set_title("circle(smoothed)", fontsize = fontsize)


  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 1].set_title("ellipse(smoothed)", fontsize = fontsize)

  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 2].set_title("ramp(smoothed)", fontsize = fontsize)

  plt.suptitle(f"testing method={method}, confset method={temp}, alpha={alpha}")
  plt.show()



# Error Check Plotting
def error_check_plot_single(sim_num, mode, shape, shape_spec, c, dim, ax, c_marg=0.2,
                                      tail="two", alpha=0.05, alpha0=0.05/4, alpha1=0.05/2):
  """
  plots error rate simulation

  Parameters
  ----------
  sim_num : int
    simulation number
  mode : str
    options for error rate "FDR" or "FNDR"
  method : str
    "BH" or "Adaptive"
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
  alpha0 : int
    [0, 1] alpha level for adaptive first stage
  alpha1 : int
    [0, 1] alpha level for adaptive second stage


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
  tbl_mth1_BH = error_check_sim_table(sim_num=sim_num, temp="1", mode=mode, method="BH",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, tail=tail)
  tbl_mth2_BH = error_check_sim_table(sim_num=sim_num, temp="2", mode=mode, method="BH",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                    alpha=alpha, tail=tail)
  tbl_mth1_AD = error_check_sim_table(sim_num=sim_num, temp="1", mode=mode, method="adaptive",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, alpha0=alpha0, alpha1=alpha1, tail=tail)
  tbl_mth2_AD = error_check_sim_table(sim_num=sim_num, temp="2", mode=mode, method="adaptive",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, alpha0=alpha0, alpha1=alpha1, tail=tail)
  method1_BH = np.mean(tbl_mth1_BH, axis=1)
  method2_BH = np.mean(tbl_mth2_BH, axis=1)
  method1_adaptive = np.mean(tbl_mth1_AD, axis=1)
  method2_adaptive = np.mean(tbl_mth2_AD, axis=1)
  ys = [method1_BH, method2_BH, method1_adaptive, method2_adaptive]
  names = ['method1_BH', 'method2_BH', 'method1_adaptive', 'method2_adaptive']

  #m0/m
  #_, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
  #m = np.sum(mu>2)
  #m0 = list()
  #for thres in c:
  # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))


  for i, y in enumerate(ys):
    ax.plot(c, y, label=names[i])
  #ax.plot(c, [alpha]*len(c), label=f"alpha={alpha}")


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
                                  dim=dim_50, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
          ax.set_title(f"{shape}, dim={dim_50}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "fdr":
            ax.set_ylim([0, 0.02])
          elif mode == "fndr":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


  # 100*100
  shape_specs_100 = shape_spec[1]
  fig, axs = plt.subplots(len(shape_specs_100), 3, figsize=figsize)
  for i in range(len(shape_specs_100)):
      for j, shape in enumerate(shapes):
          ax = axs[i, j]
          error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_100[i][j], c=c,
                                  dim=dim_100, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
          ax.set_title(f"{shape}, dim={dim_100}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "fdr":
            ax.set_ylim([0, 0.02])
          elif mode == "fndr":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


def error_check_temp(temp, mode, dim, threshold, method, shape, std=None, shape_spec=None, alpha=0.05, tail="two"):
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
    lower, upper, Achat, all_sets, n_rej = fdr_cope_temp1(data, method=method, threshold=threshold, alpha=alpha, tail=tail)
  elif temp == "2":
    lower, upper, Achat, all_sets, n_rej = fdr_cope_temp2(data, method=method, threshold=threshold, alpha=alpha, tail=tail)

  if n_rej == 0:
    ERR = 0
    #return(ERR, lower, upper, Ac)
    return(ERR)

  if temp == "0" or temp == "1" :
    ERR = -1

    if mode == "fdr":
      if tail == "one":
        numer = np.sum(np.maximum(upper - Ac.astype(int), 0))
        denom = np.sum(upper)

      elif tail == "two":
        numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
        denom = np.sum(np.minimum(upper + (1-lower), 1))


    elif mode == "fndr":
      if tail == "one":
        numer = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        denom = np.sum(1-upper)

      elif tail == "two":
        numer = np.sum(np.minimum(np.maximum(Ac.astype(int) - upper, 0) + np.maximum(lower - Ac.astype(int), 0), 1))
        #denom = np.sum(np.minimum( (1-upper) + lower, 1))
        denom = dim[1] * dim[2]

    if denom == 0:
      ERR = 0
    else:
      ERR = numer/denom

    #return(ERR, lower, upper, Ac)
    return(ERR)


  elif temp == "2" :
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
        denom2 = np.sum(1-lower)
      if denom1 == 0:
        ERR1 = 0
      else:
        ERR1 = numer1/denom1

      if denom2 == 0:
        ERR2 = 0
      else:
        ERR2 = numer2/denom2
      return( ERR1+ERR2 )

    elif mode == "fndr":
      if tail == "one":
        numer1 = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        denom1 = np.sum(1-upper)
        numer2 = 0
        denom2 = 1

      elif tail == "two":
        numer1 = np.sum(np.maximum(Ac.astype(int) - upper, 0))
        numer2 = np.sum(np.maximum(lower - Ac.astype(int) , 0))
        denom1 = np.sum(1-upper)
        denom2 = np.sum(lower)

      if denom1 == 0:
        ERR1 = 0
      else:
        ERR1 = numer1/denom1

      if denom2 == 0:
        ERR2 = 0
      else:
        ERR2 = numer2/denom2
      return( (ERR1+ERR2)/2 )

def error_check_sim_table(sim_num, temp, mode, method, shape, shape_spec, c, dim, c_marg=0.2, tail="two", alpha=0.05,
                            alpha0=0.05 / 4, alpha1=0.05 / 2):
    """
    produces table for FDR, and FNDR simulation result

    Parameters
    ----------
    sim_num : int
      simulation number
    temp : str
      options for creating confidence set "0", "1" or "2"
    mode : str
      options for error rate "FDR" or "FNDR"
    method : str
      "BH" or "Adaptive"
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
    tail : str
      "one" or "two"
    alpha : int
      [0, 1] alpha level
    alpha0 : int
      [0, 1] alpha level for adaptive first stage
    alpha1 : int
      [0, 1] alpha level for adaptive second stage

    Returns
    -------
    sim_table : array
      simulated error rate result

    Examples
    --------
    error_check_sim_table(sim_num=sim_num, temp="1", mode=mode, method="BH",
                                      shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=0.2,
                                      tail="two", alpha=0.05, alpha0=0.05/4, alpha1=0.05/2)

    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """
    sim_table = np.empty([len(c), sim_num])
    for jidx, j in enumerate(c):
      sim_temp = list()
      for i in np.arange(sim_num):
        sim_temp.append(error_check_temp(temp=temp, mode=mode, dim=dim, threshold=j, shape=shape, method=method,
                                         shape_spec=shape_spec,
                                         alpha=alpha, tail=tail))
      sim_table[jidx, :] = sim_temp
    return sim_table


def error_check_plot_single(sim_num, mode, shape, shape_spec, c, dim, ax, c_marg=0.2,
                                      tail="two", alpha=0.05, alpha0=0.05/4, alpha1=0.05/2):
  """
  plots error rate simulation

  Parameters
  ----------
  sim_num : int
    simulation number
  mode : str
    options for error rate "FDR" or "FNDR"
  method : str
    "BH" or "Adaptive"
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
  alpha0 : int
    [0, 1] alpha level for adaptive first stage
  alpha1 : int
    [0, 1] alpha level for adaptive second stage


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
  tbl_mth1_BH = error_check_sim_table(sim_num=sim_num, temp="1", mode=mode, method="BH",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, tail=tail)
  tbl_mth2_BH = error_check_sim_table(sim_num=sim_num, temp="2", mode=mode, method="BH",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                    alpha=alpha, tail=tail)
  tbl_mth1_AD = error_check_sim_table(sim_num=sim_num, temp="1", mode=mode, method="adaptive",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, alpha0=alpha0, alpha1=alpha1, tail=tail)
  tbl_mth2_AD = error_check_sim_table(sim_num=sim_num, temp="2", mode=mode, method="adaptive",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha, alpha0=alpha0, alpha1=alpha1, tail=tail)
  method1_BH = np.mean(tbl_mth1_BH, axis=1)
  method2_BH = np.mean(tbl_mth2_BH, axis=1)
  method1_adaptive = np.mean(tbl_mth1_AD, axis=1)
  method2_adaptive = np.mean(tbl_mth2_AD, axis=1)
  ys = [method1_BH, method2_BH, method1_adaptive, method2_adaptive]
  names = ['method1_BH', 'method2_BH', 'method1_adaptive', 'method2_adaptive']

  #m0/m
  #_, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
  #m = np.sum(mu>2)
  #m0 = list()
  #for thres in c:
  # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))


  for i, y in enumerate(ys):
    ax.plot(c, y, label=names[i])
  #ax.plot(c, [alpha]*len(c), label=f"alpha={alpha}")


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
                                  dim=dim_50, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
          ax.set_title(f"{shape}, dim={dim_50}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "fdr":
            ax.set_ylim([0, 0.02])
          elif mode == "fndr":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


  # 100*100
  shape_specs_100 = shape_spec[1]
  fig, axs = plt.subplots(len(shape_specs_100), 3, figsize=figsize)
  for i in range(len(shape_specs_100)):
      for j, shape in enumerate(shapes):
          ax = axs[i, j]
          error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_100[i][j], c=c,
                                  dim=dim_100, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
          ax.set_title(f"{shape}, dim={dim_100}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "fdr":
            ax.set_ylim([0, 0.02])
          elif mode == "fndr":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


def conf_plot_agg_temp(threshold, temp, method, seed=None, r=0.5, std = 5, fwhm_noise=3, fwhm_signal=20, mag = 3, fontsize = 25, figsize=(30, 20), alpha=0.05):

  """
  plots FDR controlling confidence sets for six different random fields

  Parameters
  ----------
  threshold : int
      threshold c
  temp : str
      options for creating confidence set "0", "1" or "2"
  method : str
      "BH" or "adaptive"
  r : int
      radii of ellipses
  std : int
      standard deviation for the noise field N(0, std^2)
  mag : int
      magnitude of the signal
  fontsize : int
  f   ont size for figure
  figsize : tuple
      figure size
  alpha : int
  [0, 1] alpha level

  Examples
  --------
  conf_plot_agg_temp(threshold=2, temp = "1", std=7, method="BH", _min=0, _max=3, fontsize=10, alpha=0.05, figsize = (5,3))

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  if temp == "1":
    fdr_cope_function = fdr_cope_temp1
  elif temp == "2":
    fdr_cope_function = fdr_cope_temp2
  elif temp == "0":
    fdr_cope_function = fdr_cope

  cmap1 = colors.ListedColormap(['black', 'blue'])
  cmap2 = colors.ListedColormap(['none', 'yellow'])
  cmap22 = colors.ListedColormap(['black', 'yellow'])
  cmap3 = colors.ListedColormap(['none', 'red'])
  dim_100 = (80,100,100)


  spec_cir_100_smth = {'a':r, 'b':r, 'std':std, 'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_elp_100_smth = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_signal}
  spec_ramp_100_smth = {'direction':1, 'std':std, 'mag':(0,mag), 'fwhm_noise':fwhm_noise}

  circular_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_cir_100_smth)
  ellipse_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ellipse", shape_spec=spec_elp_100_smth)
  ramp_100_smth, _ = gen_2D(dim_100, seed=seed, shape="ramp", shape_spec=spec_ramp_100_smth)


  cmap = colors.ListedColormap(['black', 'blue', 'yellow', 'red'])
  fig, axs = plt.subplots(1,3, figsize=figsize)

  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two",  alpha=alpha,  threshold=threshold)[0], cmap=cmap1)
  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[2], cmap=cmap2)
  im = axs[0].imshow(fdr_cope_function(data=circular_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[1], cmap=cmap3)
  axs[0].set_title("circle", fontsize = fontsize)

  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[2], cmap=cmap2)
  im = axs[1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, tail="two", alpha=alpha,threshold=threshold)[1], cmap=cmap3)
  axs[1].set_title("ellipse", fontsize = fontsize)

  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, tail="two", method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  im = axs[2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, tail="two", alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  axs[2].set_title("ramp", fontsize = fontsize)

  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha,  threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 0].imshow(fdr_cope_function(data=circular_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 0].set_title("circle(smoothed)", fontsize = fontsize)


  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 1].imshow(fdr_cope_function(data=ellipse_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 1].set_title("ellipse(smoothed)", fontsize = fontsize)

  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[0], cmap=cmap1)
  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[2], cmap=cmap2)
  # im = axs[1, 2].imshow(fdr_cope_function(data=ramp_100_smth, method=method, alpha=alpha, threshold=threshold)[1], cmap=cmap3)
  # axs[1, 2].set_title("ramp(smoothed)", fontsize = fontsize)

  plt.suptitle(f"testing method={method}, confset method={temp}, alpha={alpha}")
  plt.show()


dim_100 = (80,100,100)
dim_50 = (80, 50, 50)


spec_50, spec_100 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=5, mag=4, r=0.5)
spec_50_sig20, spec_100_sig20 = gen_spec(fwhm_sig=20, fwhm_noise=0, std=5, mag=4, r=0.5)
spec_50_noise10, spec_100_noise10 = gen_spec(fwhm_sig=10, fwhm_noise=10, std=5, mag=4, r=0.5)
spec_50_noise20, spec_100_noise20 = gen_spec(fwhm_sig=10, fwhm_noise=20, std=5, mag=4, r=0.5)

# for plot functions
shape_specs_50 = [spec_50, spec_50_sig20, spec_50_noise10, spec_50_noise20]
shape_specs_100 = [spec_100, spec_100_sig20, spec_100_noise10, spec_100_noise20]
shape_specs_sim = [shape_specs_50, shape_specs_100]

# for signal plotting
spec_cir_50, spec_elp_50, spec_ramp_50 = spec_50
spec_cir_100, spec_elp_100, spec_ramp_100 = spec_100

spec_cir_50_smth, spec_elp_50_smth, spec_ramp_50_smth = spec_50_noise10
spec_cir_100_smth, spec_elp_100_smth, spec_ramp_100_smth = spec_100_noise10