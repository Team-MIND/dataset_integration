"""fMRI ADHD-Autism NDA Dataset in torch."""

from bids_dataset import TorchBIDS
import re
import pandas as pd


class FMRI_AA(TorchBIDS):

  # TODO: Locate labels in dataset

  def __init__(self, root_dir, search_path, allow_multiple_files=False, 
    transforms=None, classes=None):
    """Inherits from TorchBIDS to load fMRI adhd-autism NDA dataset."""

    super(FMRI_AA, self).__init__(root_dir, search_path, scan_type="nii",
      allow_multiple_files=allow_multiple_files, transforms=transforms,
      classes=classes)

  def _get_label_map(self):
    """Load mapping between labels of all desired subjects and their
    corresponding class labels."""

    df = pd.read_csv(f"{self.root_dir}/../adhdrs01.txt", sep="\t")


def main():
  dataset = FMRI_AA("../fmri_dataset_adhd_autism/imagingcollection01",
    r'.*')

if __name__ == "__main__":
  main()