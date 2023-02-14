import numpy as np

def conf_plot(mu_set, noise_set, field_dim, threshold, alpha=0.05):
  noise = get_noise(noise_set, np.array(field_dim))
  mu = get_mu(mu_set, np.array(field_dim))
  data = noise + mu
  outer, inner, Achat, _, nrej = fdr_cope(data, threshold=threshold, alpha=alpha, tail="two")
  plot = outer + inner + (mu[0,:,:]==threshold)
  return(plot)

  def conf_plot_agg(c, _min=0, _max=3, figsize=(30, 20)):
      fig, axs = plt.subplots(2, 4, figsize=figsize)

      im = axs[0, 0].imshow(conf_plot(mu_spec_circle_l_50, noise_spec_homogen, field_dim=(80, 50, 50),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[0, 0].set_title("large circle (50*50)")

      im = axs[0, 1].imshow(conf_plot(mu_spec_circle_s_50, noise_spec_homogen, field_dim=(80, 50, 50),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[0, 1].set_title("small circle (50*50)")

      im = axs[0, 2].imshow(conf_plot(mu_spec_circle_l_100, noise_spec_homogen, field_dim=(80, 100, 100),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[0, 2].set_title("large circle (100*100)")

      im = axs[0, 3].imshow(conf_plot(mu_spec_circle_s_100, noise_spec_homogen, field_dim=(80, 100, 100),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[0, 3].set_title("small circle (100*100)")

      im = axs[1, 0].imshow(conf_plot(mu_spec_circle_l_50, noise_spec_homogen_smth, field_dim=(80, 50, 50),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[1, 0].set_title("large circle (50*50, smoothed noise)")

      im = axs[1, 1].imshow(conf_plot(mu_spec_circle_s_50, noise_spec_homogen_smth, field_dim=(80, 50, 50),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[1, 1].set_title("small circle (50*50, smoothed noise)")

      im = axs[1, 2].imshow(conf_plot(mu_spec_circle_l_100, noise_spec_homogen_smth, field_dim=(80, 100, 100),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max, )
      axs[1, 2].set_title("large circle (100*100, smoothed noise)")

      im = axs[1, 3].imshow(conf_plot(mu_spec_circle_s_100, noise_spec_homogen_smth, field_dim=(80, 100, 100),
                                      threshold=c, alpha=0.05), vmin=_min, vmax=_max)
      axs[1, 3].set_title("small circle (100*100, smoothed noise)")

      cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.5])
      fig.colorbar(im, cax=cbar_ax)

      plt.show()