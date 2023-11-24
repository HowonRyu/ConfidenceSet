import matplotlib.pyplot as plt
from .test import *



## threshold simulation
def error_check_plot_single(sim_num, mode, shape, shape_spec, c, dim, ax, c_marg=0.2, alpha=0.05):
  """
  plots error rate simulation

  Parameters
  ----------
  sim_num : int
    simulation number
  mode : str
    options for error rate "FDR" or "FNDR"
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

  tbl_joint = error_check_sim_table(sim_num=sim_num, mode=mode, method="joint",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha*2)
  tbl_separate_adaptive_lower, tbl_separate_adaptive_upper = error_check_sim_table(sim_num=sim_num, mode=mode, method="separate_adaptive",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha)
  tbl_separate_BH_lower, tbl_separate_BH_upper = error_check_sim_table(sim_num=sim_num, mode=mode, method="separate_BH",
                                    shape=shape, shape_spec=shape_spec, c=c, dim=dim, c_marg=c_marg,
                                     alpha=alpha)
  #tbl_separate_avg = (tbl_separate_lower + tbl_separate_upper)/2


  joint = np.mean(tbl_joint, axis=1)
  separate_upper_BH = np.mean(tbl_separate_BH_upper, axis=1)
  separate_lower_adaptive = np.mean(tbl_separate_adaptive_lower, axis=1)
  separate_lower_BH = np.mean(tbl_separate_BH_lower, axis=1)
  #separate_avg = np.mean(tbl_separate_avg, axis=1)

  ys = [joint, separate_lower_adaptive, separate_lower_BH, separate_upper_BH]
  names = ['Joint', 'Separate(lower adaptive)', 'Separate(lower BH)', 'Separate(upper BH)']

  #m0/m
  #_, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
  #m = np.sum(mu>2)
  #m0 = list()
  #for thres in c:
  # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))


  for i, y in enumerate(ys):
    ax.plot(c, y, label=names[i])


def error_check_plot(sim_num, c, mode, shape_spec, c_marg=0.2,  alpha=0.05, alpha0=0.05/4, alpha1=0.05/2, figsize=(15,10)):
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
                                  dim=dim_50, ax=ax, c_marg=c_marg, alpha=alpha)
          ax.set_title(f"{shape}, dim={dim_50}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
          ax.set_xlabel("threshold")
          ax.set_ylabel(str(mode))
          if mode == "FDR":
            ax.set_ylim([0, 0.07])
          elif mode == "FNDR":
            ax.set_ylim([0,1])
          ax.legend()
  plt.show()


  # 100*100
  #shape_specs_100 = shape_spec[1]
  #fig, axs = plt.subplots(len(shape_specs_100), 3, figsize=figsize)
  #for i in range(len(shape_specs_100)):
  #    for j, shape in enumerate(shapes):
  #        ax = axs[i, j]
  #        error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_100[i][j], c=c,
  #                                dim=dim_100, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
  #        ax.set_title(f"{shape}, dim={dim_100}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
  #        ax.set_xlabel("threshold")
  #        ax.set_ylabel(str(mode))
  #        if mode == "fdr":
  #          ax.set_ylim([0, 0.02])
  #        elif mode == "fndr":
  #          ax.set_ylim([0,1])
  #        ax.legend()
  #plt.show()



## noise fwhm simulation
def sim_table_noise(sim_num, mode, method, shape, shape_spec, threshold, noise_fwhm, dim, alpha=0.05):
    """
    produces table for FDR, and FNDR simulation result per different noise smoothness level

    Parameters
    ----------
    sim_num : int
      simulation number
    mode : str
      options for error rate "FDR" or "FNDR"
    method : str
      "joint", "separate_adaptive" or "separate_BH"
    shape : str
      "ramp" or "ellipse"
    shape_spec : dict
      dictionary containing shape specs - specify the shape spec except for shape_spec['fwhm_noise']
    threshold : int
      threshold level
    noise_fwhm : int
      list of full width half maximum for noise smoothing
    dim : int
      dimension of the image (N, W, H)
    alpha : int
      [0, 1] alpha level

    Returns
    -------
    sim_table : array
      simulated error rate result

    Examples
    --------
    sim_table_noise(sim_num=sim_num, mode=mode, method="joint", noise_fwhm=[3,5,6,7,8]
                                      shape=shape, shape_spec=shape_spec, threshold=2,
                                      dim=dim,  alpha=0.05)

    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """

    if method == "separate_adaptive" or method == "separate_BH":
        sim_table_lower = np.empty([len(noise_fwhm), sim_num])
        sim_table_upper = np.empty([len(noise_fwhm), sim_num])

        for jidx, j in enumerate(noise_fwhm):
            sim_temp_upper = list()
            sim_temp_lower = list()
            shape_spec['fwhm_noise'] = j
            for i in np.arange(sim_num):
                sim_temp_upper.append(error_check(mode=mode, dim=dim, threshold=threshold, shape=shape,
                                                  method=method, shape_spec=shape_spec, alpha=alpha)[1])
                sim_temp_lower.append(error_check(mode=mode, dim=dim, threshold=j, shape=shape,
                                                  method=method, shape_spec=shape_spec, alpha=alpha)[0])
            sim_table_lower[jidx, :] = sim_temp_lower
            sim_table_upper[jidx, :] = sim_temp_upper
        return sim_table_lower, sim_table_upper

    if method == "joint":
        sim_table = np.empty([len(noise_fwhm), sim_num])
        for jidx, j in enumerate(noise_fwhm):
            sim_temp = list()
            shape_spec['fwhm_noise'] = j
            for i in np.arange(sim_num):
                sim_temp.append(error_check(mode=mode, dim=dim, threshold=threshold, shape=shape,
                                            method=method, shape_spec=shape_spec, alpha=alpha))
            sim_table[jidx, :] = sim_temp
        return sim_table


def sim_plot_single_noise(sim_num, mode, shape, shape_spec, threshold,
                          noise_fwhm, dim, ax, alpha=0.05):
    """
    plots error rate simulation

    Parameters
    ----------
      sim_num : int
        simulation number
      mode : str
        options for error rate "FDR" or "FNDR"
      method : str
        "joint", "separate_adaptive" or "separate_BH"
      shape : str
        "ramp" or "ellipse"
      shape_spec : dict
        dictionary containing shape specs - specify the shape spec except for shape_spec['fwhm_noise']
      threshold : int
        threshold level
      noise_fwhm : int
        list of full width half maximum for noise smoothing
      dim : int
        dimension of the image (N, W, H)
      alpha : int
        [0, 1] alpha level
    Examples
    --------
    noise_fwhm = [2,4,6,8,10]
    sim_plot_single_noise(sim_num=sim_num, mode=mode, shape=shape,
                                  shape_spec=shape_specs_50[i][j], threshold=threshold,
                                  noise_fwhm=noise_fwhm, dim=dim_50, ax=ax, alpha=alpha)
    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """

    tbl_joint = sim_table_noise(sim_num=sim_num, mode=mode, method="joint",
                                shape=shape, shape_spec=shape_spec, threshold=threshold, dim=dim,
                                alpha=alpha * 2, noise_fwhm=noise_fwhm)
    tbl_separate_adaptive_lower, tbl_separate_adaptive_upper = sim_table_noise(sim_num=sim_num, mode=mode,
                                                                               method="separate_adaptive",
                                                                               shape=shape, shape_spec=shape_spec,
                                                                               threshold=threshold, dim=dim,
                                                                               alpha=alpha, noise_fwhm=noise_fwhm)
    tbl_separate_BH_lower, tbl_separate_BH_upper = sim_table_noise(sim_num=sim_num, mode=mode, method="separate_BH",
                                                                   shape=shape, shape_spec=shape_spec,
                                                                   threshold=threshold, dim=dim,
                                                                   alpha=alpha, noise_fwhm=noise_fwhm)
    # tbl_separate_avg = (tbl_separate_lower + tbl_separate_upper)/2

    joint = np.mean(tbl_joint, axis=1)
    separate_upper_BH = np.mean(tbl_separate_BH_upper, axis=1)
    separate_lower_adaptive = np.mean(tbl_separate_adaptive_lower, axis=1)
    separate_lower_BH = np.mean(tbl_separate_BH_lower, axis=1)
    # separate_avg = np.mean(tbl_separate_avg, axis=1)

    ys = [joint, separate_lower_adaptive, separate_lower_BH, separate_upper_BH]
    names = ['Joint', 'Separate(lower adaptive)', 'Separate(lower BH)', 'Separate(upper BH)']

    # m0/m
    # _, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
    # m = np.sum(mu>2)
    # m0 = list()
    # for thres in c:
    # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))

    for i, y in enumerate(ys):
        ax.plot(noise_fwhm, y, label=names[i])


def sim_plot_noise(sim_num, noise_fwhm, mode, threshold=2, std=5, mag=4,
                   alpha=0.05, alpha0=0.05 / 4, alpha1=0.05 / 2,
                   font_size=9, figsize=(15, 10)):
    """
    combines error_check_plot_single to create a grid of simulations plots with different simulation settings

    Parameters
    ----------
    sim_num : int
      simulation number
    threshold : int
      threshold level
    noise_fwhm : int
      list of full width half maximum for noise smoothing
    mode : str
      options for error rate "FDR" or "FNDR"
    shape_spec : dict
      dictionary containing shape specs
    figsize : tuple
      figure size
    std : int
      standard deviation for the noise field N(0, std^2)
    mag : int
      magnitude of the signal

    Examples
    --------
    error_check_plot(sim_num=100, mode="fdr", c=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], shape_spec=shape_specs_sim, figsize=(23,30))

    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """
    # simulation setting
    shapes = ["circular", "ellipse", "ramp"]
    dim_50 = (80, 50, 50)

    spec_50_sig5, spec_100_sig5 = gen_spec(fwhm_sig=5, fwhm_noise=0, std=std, mag=mag, r=0.5)
    spec_50_sig10, spec_100_sig10 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=std, mag=mag, r=0.5)

    shape_specs_50 = [spec_50_sig5, spec_50_sig10]

    # 50*50
    fig, axs = plt.subplots(len(shape_specs_50), 3, figsize=figsize)
    for i in range(len(shape_specs_50)):
        for j, shape in enumerate(shapes):
            ax = axs[i, j]
            sim_plot_single_noise(sim_num=sim_num, mode=mode, shape=shape,
                                  shape_spec=shape_specs_50[i][j], threshold=threshold,
                                  noise_fwhm=noise_fwhm, dim=dim_50, ax=ax, alpha=alpha)
            ax.set_title(
                f"{shape}, c={threshold}, signal fwhm={shape_specs_50[i][j]['fwhm_signal']}, std={shape_specs_50[i][j]['std']}")  # dim={dim_50},
            ax.title.set_fontsize(font_size)
            ax.set_xlabel("noise smoothing (fwhm)")
            ax.set_ylabel(str(mode))
            # if mode == "FDR":
            #  ax.set_ylim([0, 0.07])
            # elif mode == "FNDR":
            #  ax.set_ylim([0,1])
            ax.legend()
    plt.show()

    # 100*100
    #  dim_100 = (80,100,100)
    # shape_specs_100 = [spec_100_sig10, spec_100_sig10]
    # fig, axs = plt.subplots(len(shape_specs_100), 3, figsize=figsize)
    # for i in range(len(shape_specs_100)):
    #    for j, shape in enumerate(shapes):
    #        ax = axs[i, j]
    #        error_check_plot_single(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_specs_100[i][j], c=c,
    #                                dim=dim_100, ax=ax, c_marg=c_marg, tail=tail, alpha=alpha, alpha0=alpha0, alpha1=alpha1)
    #        ax.set_title(f"{shape}, dim={dim_100}, fwhm_noise={ shape_specs_50[i][j]['fwhm_noise'] }, fwhm_signal={ shape_specs_50[i][j]['fwhm_signal']}") #, std={ shape_specs_100[i][j]['std'] }
    #        ax.set_xlabel("threshold")
    #        ax.set_ylabel(str(mode))
    #        if mode == "fdr":
    #          ax.set_ylim([0, 0.02])
    #        elif mode == "fndr":
    #          ax.set_ylim([0,1])
    #        ax.legend()
    # plt.show()



  ## signal fwhm simulation







