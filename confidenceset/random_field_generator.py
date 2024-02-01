import numpy as np

from scipy.ndimage import gaussian_filter


def gen_spec(fwhm_sig=10, fwhm_noise=5, std=5, mag=4, r=0.5):
  """
  generates dictionaries of the shape spec sets to be used in random field generator

  Parameters
  ----------
  fwhm_sig : float
    full width at half maximum for signal
  fwhm_noise : float
    full width at half maximum for noise field
  std : float
    standard deviation for the noise field N(0, std^2)
  mag : float
    magnitude of the signal
  r : float
    radii of ellipses

  Returns
  -------
  spec_dic_set_50 : list
    list of dictionaries for image dimension 50*50
  spec_dic_set_100 : list
    list of dictionaries for image dimension 100*100

  Examples
  --------
  spec_50, spec_100 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=5, mag=4, r=0.5)
  gen_2D(80, 50, 50)), shape="ellipse", shape_spec=spec_50[0])

  :Authors:
    Howon Ryu <howonryu@ucsd.edu>
  """
  cir_50 = {'a':r, 'b':r, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':fwhm_sig/2}
  elp_50 = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':fwhm_sig/2}
  ramp_50 = {'direction':1, 'std':std,'mag':(0,mag), 'fwhm_noise':fwhm_noise/2, 'fwhm_signal':0}
  spec_dic_set_50 = [cir_50, elp_50, ramp_50]

  cir_100 = {'a':r, 'b':r, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_sig}
  elp_100 = {'a':r*2, 'b':r*0.5, 'std':std,'mag':mag, 'fwhm_noise':fwhm_noise, 'fwhm_signal':fwhm_sig}
  ramp_100 = {'direction':1, 'std':std,'mag':(0,mag), 'fwhm_noise':fwhm_noise, 'fwhm_signal':0}
  spec_dic_set_100 = [cir_100, elp_100, ramp_100]

  return(spec_dic_set_50, spec_dic_set_100)




def gen_2D(dim, shape, shape_spec, seed=None, truncate=3):
  """
  generates 2D image

  Parameters
  ----------
  dim : tuple
    dimension of the image (N, W, H)
  shape : str
    shape of the signal; choose from ramp, circle or step. The rest is automatically ellipse.
  shape_spec : dict
    dictionary storing shape parameters
  seed : int
    set seed for signals
  truncate : float
    truncation point for the smoothing

  Returns
  -------
  data : array
    generated 2D field
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  spec_50, spec_100 = gen_spec(fwhm_sig=10, fwhm_noise=0, std=5, mag=4, r=0.5)
  gen_2D((80, 50, 50), shape="ellipse", shape_spec=spec_50[0])
  """
  fwhm_noise = shape_spec['fwhm_noise']
  std = shape_spec['std']
  nsubj = dim[0]
  mu = np.zeros(dim)

  # signal
  if shape == "step":
    mu = step_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  elif shape == "ramp":
    mu = ramp_2D(dim=dim, shape_spec=shape_spec)
  elif shape == "circle":
    mu = circle_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  else:
    mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)

  # noise
  np.random.seed(seed)

  noise = np.random.randn(*dim) * std
  sigma_noise = fwhm_noise / np.sqrt(8 * np.log(2))

  # apply Gaussian filter to the field (subject by subject)
  for i in np.arange(nsubj):
    noise[i,:,:] = gaussian_filter(noise[i,:,:], sigma = sigma_noise, truncate=truncate)  #smoothing

  sig = np.zeros((dim[1],dim[2]))
  sig[int(dim[1]/2), int(dim[2]/2)] = 1
  kernel = gaussian_filter(sig, sigma = sigma_noise, truncate=truncate) # retreived the Gaussian kernel used for smoothing

  scale = np.sqrt(np.sum(kernel**2)) # scale the smoothed noise field by sqrt(sum(kernel^2))


  data = np.array(mu + noise/scale, dtype='float')
  return(data, mu)


def ramp_2D(dim, shape_spec):
  """
  generates 2D ramp signal

  Parameters
  ----------
  dim : tuple
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters including direction, mag, and std

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = ramp_2D(dim=dim, shape_spec=shape_spec)
  """
  nsubj = dim[0]
  direction = shape_spec['direction']
  mag = shape_spec['mag']
  std = shape_spec['std']

  # signal
  if direction == 0: #vertical
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2])[::-1],dim[1]).reshape(dim[1],dim[2])
  if direction == 1: #horizontal
    mu_temp = np.repeat(np.linspace(mag[0], mag[1], dim[2]),dim[1]).reshape(dim[2],dim[1]).transpose()
  mu = np.array(mu_temp, dtype='float')
  return(mu)

def ellipse_2D(dim, shape_spec, truncate=3):
  """
  generates 2D ellipse signal

  Parameters
  ----------
  dim : tuple
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters including a, b, mag, std, and fwhm_signal
  truncate : float
    truncation point for the smoothing

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = ellipse_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  """

  nsubj = dim[0]
  a = shape_spec['a']
  b = shape_spec['b']
  mag = shape_spec['mag']
  fwhm_signal = shape_spec['fwhm_signal']

  # signal
  x, y = np.meshgrid(np.linspace(-1,1,dim[1]), np.linspace(-1,1,dim[2]))
  cx, cy = 0,0
  theta = -np.pi/4
  xx = np.cos(theta)*(x-cx) + np.sin(theta)*(y-cy)
  yy = -np.sin(theta)*(x-cx) + np.cos(theta)*(y-cy)
  ellipse = np.array((xx/a)**2 + (yy/b)**2 <= 1, dtype="float") * mag

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  ellipse_smth = gaussian_filter(ellipse, sigma = sigma_signal, truncate=truncate)
  mu = np.array(ellipse_smth, dtype='float')

  return(mu)

def circle_2D(dim, shape_spec, truncate=3):
  """
  generates 2D circle signal

  Parameters
  ----------
  dim : tuple
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters
  truncate : float
    truncation point for the smoothing

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = circle_2D(dim=(80,50,50), shape_spec=shape_spec, truncate=3)
  """
  nsubj = dim[0]
  mag = shape_spec['mag']
  fwhm_signal = shape_spec['fwhm_signal']
  r = shape_spec['r']

  # signal
  x = np.linspace(-1, 1, dim[1])
  y = np.linspace(-1, 1, dim[2])
  xx, yy = np.meshgrid(x, y) #grid

  circle = np.array(xx**2 + yy**2 <= r**2, dtype="float") * mag

  # smoothing
  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  circle_smth = gaussian_filter(circle, sigma = sigma_signal, truncate=truncate)
  mu = np.array(circle_smth, dtype='float')
  return mu

def step_2D(dim, shape_spec, truncate=3):
  """
  generates 2D step signal

  Parameters
  ----------
  dim : tuple
    dimension of the image (N, W, H)
  shape_spec : dict
    dictionary storing shape parameters
  truncate : float
    truncation point for the smoothing

  Returns
  -------
  mu : array
    generated 2D signal (mu) field

  Examples
  --------
  mu = step_2D(dim=dim, shape_spec=shape_spec, truncate=truncate)
  """
  fwhm_signal = shape_spec['fwhm_signal']

  # signal
  step = np.zeros((dim[1], dim[2]))
  step[:, int(dim[2]/2)+1:] = 1
  step[:, :int(dim[2]/2)+1] = -1
  step = step.astype(float)

  sigma_signal = fwhm_signal / np.sqrt(8 * np.log(2))
  mu = gaussian_filter(step, sigma=sigma_signal, truncate=truncate)
  return mu


