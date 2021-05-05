"""Put custom (composed) tensor transformations here."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


################################################################################
# Single Transformations
################################################################################

class RandomFlips(object):

  def __init__(self, x_flips = True, y_flips = True, z_flips = True):
    """Randomly flip a 3d tensor on all axes."""
    self.x_flips = x_flips
    self.y_flips = y_flips
    self.z_flips = z_flips

  def __call__(self, x):

    if np.random.uniform() > .5 and self.x_flips:
      x = torch.flip(x, dims = [0])

    if np.random.uniform() > .5 and self.y_flips:
      x = torch.flip(x, dims = [1])

    if np.random.uniform() > .5 and self.z_flips:
      x = torch.flip(x, dims = [2])

    return x
        

class ZScore(object):
  def __init__(self, dataset=None, full_normalization=False,
    scale_factor=1.0):
    """A tensor normalization transformation.

    Performs based normalization of a tensor by subtracting
    its mean from all values and dividing by its stddev.

    Keyword Args:
        dataset: The dataset with which this transformation will be
            used. Only necessary if the full_normalization flag is
            set to true (default None).
        full_normalization: A flag indicating whether or not the tensor
            values will be normalized for just itself (False) or using
            the mean and stddev for the entire dataset (True). If True,
            when this transformation is initialized, it makes a single
            pass through the dataset. At each sample, it  adds the sum of
            all its values and the sum of all its squared
            values to a counter. After all samples are seen, these sums
            will be used to compute the new mean and stddev of the entire
            dataset, which will be used in subsequent normalizations.
        scale_factor: Float, determines by how much original scan values
            are scaled down before being used for full-dataset 
            normalization. Because determining the mean and stddev of
            large scan datasets requires adding many parameters, this
            scale factor can be tweaked to improve precision.
            Note that scan values after normalization are scaled back.
            (default 1.0)
    """

    self.full_normalization = full_normalization

    # Not used unless full_normalization flag is set
    self.ds_mean = self.ds_std = None

    if full_normalization:

      if not dataset:
        print("dataset object must be passed to Normalize() for full \
            normalization!")
        exit()

      print("Normalizing...")

      # Track sum of all tensor values and squared values
      sum_vals = 0.0
      sum_sq_vals = 0.0

      # Iterate over entire dataset
      for sample in dataset:
        scan, _ = sample

        # Scale down tensor values
        scan /= scale_factor

        # Sum scan values
        sum_vals += scan.sum()
        # Square scan values
        sq_scan = scan * scan
        # Sum squared values
        sum_sq_vals += sq_scan.sum()

      # Use total dataset vals and squared vals to obtain mean and stddev
      sample_size = dataset[0][0].numel()
      n = sample_size * len(dataset)
      ds_mean = sum_vals / n
      ds_std = math.sqrt((sum_sq_vals / n) - (ds_mean ** 2))
      # Re-adjust using scale factor
      self.ds_mean = scale_factor * ds_mean
      self.ds_std = scale_factor * ds_std
      print(f"\tDataset Mean: {self.ds_mean:.4f}")
      print(f"\tDataset Stddev: {self.ds_std:.4f}")

  def normalize(self, x):
    """Perform normalization on a single sample using its mean and stddev.
    """

    x = (x - x.mean()) / x.std()
    return x

  def __call__(self, sample):
    if self.full_normalization:
      x = (x - self.ds_mean) / self.ds_std
      return x
    else:
      return self.normalize(sample)


class Downsample(object):

  def __init__(self, size=None, scale_factor=0.5):
    """Downsamples a given tensor to the specified size using bilinear
    interpolation. For volumetric input, expects input as a 5d tensor with 
    dims as (batch_size, channels, depth, height, width). In this case,
    size is specified as a 3d tensor equal to (new_depth, new_height,
    new_width).

    Keyword Args:
      size: Tuple specifying output depth(/width/height).
      scale_factor: Alternatively, specifying a scale factor will reduce the
        size of the input tensor such that the size of each dim after
        transformation is equal to floor(scale_Factor * d) where d is the
        original size of that dim (default 0.5).
    """

    self.size = size
    if self.size:
      self.scale_factor = None
    else:
      self.scale_factor = scale_factor

  def __call__(self, x):
    return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
      recompute_scale_factor=True)

def main():
  """Downsampling test.

  :meta private:
  """
  t = Downsample()
  x = torch.ones((1, 1, 50, 50, 50))
  x = t(x)
  print(x.size())

if __name__ == "__main__":
  main()     