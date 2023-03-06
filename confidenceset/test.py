import numpy as np

def fwe_inclusion_check(n_subj, img_dim, c, noise_set, mu_set,
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


def fdr_error_check(dim, c, shape, method, shape_spec=None, mag=3, direction=1, fwhm=3,
                    std=5, alpha=0.05, tail="two"):
  if (shape == 'noise'):
    data = np.random.randn(*dim) * std
    mu = np.zeros((dim[1],dim[2]))

  if (shape == "circular"):
    data, mu = circular_2D(dim=dim, shape_spec = shape_spec)
  if (shape == "ramp"):
    data, mu = ramp_2D(dim=dim, std=std, mag=(0,mag), direction=direction, fwhm=fwhm)

  Ac = mu>=c
  AcC = 1-Ac
  lower, upper, Achat, all_sets, n_rej = fdr_cope(data, method = method, threshold=c, alpha=alpha, tail=tail)

  if tail == "one":
    numer = np.sum(np.maximum(upper - Ac.astype(int), 0))
    denom = np.sum(upper)

  if tail == "two":
    numer = np.sum(np.minimum(np.maximum(upper - Ac.astype(int), 0) + np.maximum(Ac.astype(int) - lower, 0), 1))
    denom = np.sum(np.minimum(upper + (1-lower), 1) )

  if n_rej ==0:
    ERR=0
  else:
    ERR = numer / denom

  return(ERR)


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

    # m0/m
    ERR['alpha*m0/m(small)'] = [np.round(0.05 * m0_small50_c0 / (dimprod_50), 6),
                                np.round(0.05 * m0_small50_c1 / (dimprod_50), 6),
                                np.round(0.05 * m0_small50_c2 / (dimprod_50), 6)]

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

    # m0/m
    ERR['alpha*m0/m(large)'] = [np.round(0.05 * m0_large50_c0 / (dimprod_50), 6),
                                np.round(0.05 * m0_large50_c1 / (dimprod_50), 6),
                                np.round(0.05 * m0_large50_c2 / (dimprod_50), 6)]

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

    # m0/m
    ERR['alpha*m0/m(ramp)'] = [np.round(0.05 * m0_ramp_50_c0 / (50 * 50), 6),
                               np.round(0.05 * m0_ramp_50_c1 / (50 * 50), 6),
                               np.round(0.05 * m0_ramp_50_c2 / (50 * 50), 6)]

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

    # m0/m
    ERR2['alpha*m0/m(small)'] = [np.round(0.05 * m0_small100_c0 / (dimprod_100), 6),
                                 np.round(0.05 * m0_small100_c1 / (dimprod_100), 6),
                                 np.round(0.05 * m0_small100_c2 / (dimprod_100), 6)]

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
    # m0/m
    ERR2['alpha*m0/m(large)'] = [np.round(0.05 * m0_large100_c0 / (dimprod_100), 6),
                                 np.round(0.05 * m0_large100_c1 / (dimprod_100), 6),
                                 np.round(0.05 * m0_large100_c2 / (dimprod_100), 6)]

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

    # m0/m
    ERR2['alpha*m0/m(ramp)'] = [np.round(0.05 * m0_ramp_100_c0 / (100 * 100), 6),
                                np.round(0.05 * m0_ramp_100_c1 / (100 * 100), 6),
                                np.round(0.05 * m0_ramp_100_c2 / (100 * 100), 6)]

    ERR2_key_calc = [list(ERR2.keys())[i] for i in [1, 2, 4, 5, 7, 8]]
    ERR2.update({n: np.round(np.nanmean(ERR2[n], axis=1), 4) for n in ERR2_key_calc})

    return (ERR, ERR2)