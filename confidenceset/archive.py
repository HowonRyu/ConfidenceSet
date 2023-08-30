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