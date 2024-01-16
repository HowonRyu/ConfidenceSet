import matplotlib.pyplot as plt
from .test import *



## threshold simulation
def sim_table_threshold(sim_num, mode, dim, shape, shape_spec, c,
                        c_marg=0.2, alpha=0.05, sanity_check=False):
    """
    produces table for FDR, and FNDR simulation result per threshold change

    Parameters
    ----------
    sim_num : int
      simulation number
    mode : str
      options for error rate "FDR" or "FNDR"
    dim : tuple
      dimension of the image (N, W, H)
    shape : str
      shape of the signal; choose from ramp or step. The rest is automatically ellipse.
    shape_spec : dict
      dictionary containing shape specs
    c : list
      list of thresholds (float)
    c_marg : float
      margin allowed for the threshold
    alpha : float
      [0, 1] alpha level
    sanity_check : Boolean
      sanity check with noise field (no signal)

    Returns
    -------
    sim_table : array
      simulated error rate result

    Examples
    --------
    sim_table_adaptive_lower, sim_table_BH_lower, sim_table_BH_upper, sim_table_joint = sim_table_threshold(sim_num=sim_num, mode=mode, shape=shape,
                                                                                                          shape_spec=shape_spec, c=c, dim=dim,
                                                                                                          c_marg=c_marg, alpha=alpha)
    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """

    sim_table_adaptive_lower = np.empty([len(c), sim_num])
    sim_table_BH_lower = np.empty([len(c), sim_num])
    sim_table_BH_upper = np.empty([len(c), sim_num])
    sim_table_joint = np.empty([len(c), sim_num])


    for thres_idx, thres in enumerate(c):
      temp_sim_table_adaptive_lower = list()
      temp_sim_table_BH_lower = list()
      temp_sim_table_BH_upper = list()
      temp_sim_table_joint = list()
      for i in np.arange(sim_num):
        #seed = np.random.randint(0, 2**32 - 1)
        #print(f'seed:{seed}')

        if sanity_check:
          data = np.random.randn(*dim)
          mu = np.zeros(dim[1], dim[2])
        else:
          data, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)

        separate_adaptive_error_check = error_check(mode=mode, data=data, mu=mu, threshold=thres,
                                          method="separate_adaptive",  alpha=alpha)
        separate_BH_error_check = error_check(mode=mode, data=data, mu=mu, threshold=thres,
                                          method="separate_BH", alpha=alpha)
        joint_error_check = error_check(mode=mode, data=data, mu=mu, threshold=thres,
                                          method="joint", alpha=alpha*2)

        temp_sim_table_adaptive_lower.append(separate_adaptive_error_check[0])
        temp_sim_table_BH_lower.append(separate_BH_error_check[0])
        temp_sim_table_BH_upper.append(separate_BH_error_check[1])
        temp_sim_table_joint.append(joint_error_check)

      sim_table_adaptive_lower[thres_idx, :] = temp_sim_table_adaptive_lower
      sim_table_BH_lower[thres_idx, :] = temp_sim_table_BH_lower
      sim_table_BH_upper[thres_idx, :] = temp_sim_table_BH_upper
      sim_table_joint[thres_idx, :] = temp_sim_table_joint

    return sim_table_adaptive_lower, sim_table_BH_lower, sim_table_BH_upper, sim_table_joint


def sim_plot_single_threshold(sim_num, dim, mode, shape, shape_spec, c, ax,
                              c_marg=0.2, alpha=0.05):
    """
    plots error rate simulation for one setting (use within sim_threshold)

    Parameters
    ----------
    sim_num : int
      simulation number
    dim : tuple
      dimension of the image (N, W, H)
    mode : str
      options for error rate "FDR" or "FNDR"
    shape : str
      shape of the signal; choose from ramp or step. The rest is automatically ellipse.
    shape_spec : dict
      dictionary containing shape specs
    c : list
      list of thresholds (float)
    ax : axes
      subplot figure to use
    c_marg : float
      margin allowed for the threshold
    alpha : float
      [0, 1] alpha level

    Returns
    -------
    sim_result_single : dict
      simulation result for single plot


    Examples
    --------
    dim_50 = (80,50,50)
    dim_100 = (80,100,100)
    sim_result = dict()
    # 50*50
    fig, axs = plt.subplots(len(fwhm_noise_vec), len(fwhm_signal_vec), figsize=figsize)

    for i, fwhm_noise in enumerate(fwhm_noise_vec):
        for j, fwhm_signal in enumerate(fwhm_signal_vec):
            ax = axs[i, j]
            shape_spec['fwhm_noise'] = fwhm_noise
            shape_spec['fwhm_signal'] = fwhm_signal
            sim_result_single, sim_error = sim_plot_single_threshold(sim_num=sim_num, mode=mode, shape=shape, shape_spec=shape_spec, c=c,
                                    dim=dim_50, ax=ax, c_marg=c_marg, alpha=alpha)

            # sim_result per plot
            key_name = "noise"+str(fwhm_noise)+"signal"+str(fwhm_signal)
            sim_result[key_name] = sim_result_single

            # plotting
            ax.set_title(f"fwhm(noise)={fwhm_noise}, fwhm(signal)={fwhm_signal}, \n SE={sim_error}, shape={shape}") #std={std}
            ax.set_xlabel("c")
            ax.set_ylabel(str(mode))
            if mode == "FDR":
              ax.set_ylim([0, 0.2])
            elif mode == "FNDR":
              ax.set_ylim([0,1.1])
            ax.legend()
    plt.show()


    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """
    sim_table_adaptive_lower, sim_table_BH_lower, sim_table_BH_upper, sim_table_joint = sim_table_threshold(
        sim_num=sim_num, mode=mode, shape=shape,
        shape_spec=shape_spec, c=c, dim=dim,
        c_marg=c_marg, alpha=alpha)

    sim_std_stacked = np.stack([sim_table_adaptive_lower, sim_table_BH_lower, sim_table_BH_upper, sim_table_joint],
                               axis=0)
    sim_std_error = round(np.std(sim_std_stacked) / np.sqrt(sim_num), 3)

    # tbl_separate_avg = (tbl_separate_lower + tbl_separate_upper)/2

    joint = np.mean(sim_table_joint, axis=1)
    separate_upper_BH = np.mean(sim_table_BH_upper, axis=1)
    separate_lower_adaptive = np.mean(sim_table_adaptive_lower, axis=1)
    separate_lower_BH = np.mean(sim_table_BH_lower, axis=1)
    # separate_avg = np.mean(tbl_separate_avg, axis=1)

    methods = [joint, separate_lower_adaptive, separate_lower_BH, separate_upper_BH]
    methods_names = ['Joint', 'Separate(lower adaptive)', 'Separate(lower BH)', 'Separate(upper BH)']
    method_key = ['joint', 'lower_adaptive)', 'lower_BH)', 'upper_BH']
    sim_result_single = dict(zip(method_key, methods))
    # m0/m
    # _, mu = gen_2D(dim=dim, shape=shape, shape_spec=shape_spec)
    # m = np.sum(mu>2)
    # m0 = list()
    # for thres in c:
    # m0.append(np.sum(np.logical_and(mu < thres+c_marg, mu > thres-c_marg)))

    for i, method in enumerate(methods):
        ax.plot(c, method, label=methods_names[i])
        if mode == "FDR":
            ax.axhline(y=0.05, color='red', linestyle='--')
    return sim_result_single, sim_std_error


def sim_threshold(sim_num, c, mode, shape, fwhm_signal_vec, fwhm_noise_vec, std,
                  c_marg=0.2, alpha=0.05, alpha0=0.05 / 4, alpha1=0.05 / 2, figsize=(15, 10)):
    """
    combines error_check_plot_single to create a grid of simulations plots with different simulation settings

    Parameters
    ----------
    sim_num : int
      simulation number
    c : list
      list of thresholds (float)
    mode : str
      options for error rate "FDR" or "FNDR"
    shape_spec : dict
      dictionary containing shape specs
    figsize : tuple
      figure size
    Returns
    -------
    sim_result : dict
      simulation result

    Examples
    --------
    FDR_step = sim_threshold(sim_num=500, c=np.linspace(-2, 2, num=21), mode="FDR", shape="step",
                   fwhm_signal_vec=[0,5,10], fwhm_noise_vec=[0,5,15], std=1,
                       c_marg=0.2,  alpha=0.05, alpha0=0.05/4, alpha1=0.05/2, figsize=(20,20))

    :Authors:
      Howon Ryu <howonryu@ucsd.edu>
    """
    if shape == "step":
        shape_spec = {'fwhm_signal': 0,
                      'fwhm_noise': 0,
                      'std': std}
    elif shape == "ramp":
        shape_spec = {'direction': 1,
                      'mag': (-1, 1),
                      'fwhm_noise': 0,
                      'std': std}
    elif shape == "circle":
        shape_spec = {'r':0.5, 'mag': 3,
                      'fwhm_signal': 0,
                      'fwhm_noise': 0,
                      'std': std}

    dim_50 = (80, 50, 50)
    dim_100 = (80, 100, 100)
    sim_result = dict()
    # 50*50
    fig, axs = plt.subplots(len(fwhm_noise_vec), len(fwhm_signal_vec), figsize=figsize)

    for i, fwhm_noise in enumerate(fwhm_noise_vec):
        for j, fwhm_signal in enumerate(fwhm_signal_vec):
            ax = axs[i, j]
            shape_spec['fwhm_noise'] = fwhm_noise
            shape_spec['fwhm_signal'] = fwhm_signal
            sim_result_single, sim_error = sim_plot_single_threshold(sim_num=sim_num, mode=mode, shape=shape,
                                                                     shape_spec=shape_spec, c=c,
                                                                     dim=dim_50, ax=ax, c_marg=c_marg, alpha=alpha)

            # sim_result per plot
            key_name = "noise" + str(fwhm_noise) + "signal" + str(fwhm_signal)
            sim_result[key_name] = sim_result_single

            # plotting
            ax.set_title(
                f"fwhm(noise)={fwhm_noise}, fwhm(signal)={fwhm_signal}, \n SE={sim_error}, shape={shape}")  # std={std}
            ax.set_xlabel("c")
            ax.set_ylabel(str(mode))
            if mode == "FDR":
                ax.set_ylim([0, 0.2])
            elif mode == "FNDR":
                ax.set_ylim([0, 1.1])
            ax.legend()
    plt.show()

    return sim_result








