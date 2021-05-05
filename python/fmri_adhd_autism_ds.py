"""fMRI ADHD-Autism NDA Dataset in torch."""

from neuroimaging_dataset import TorchNI
import re
import pandas as pd


class FMRI_AA(TorchNI):

  # TODO: Locate labels in dataset

  def __init__(self, root_dir, search_path, allow_multiple_files=False, 
    transforms=None, classes=None):
    """Inherits from TorchBIDS to load fMRI adhd-autism NDA dataset.
    
    See full study at https://nda.nih.gov/edit_collection.html?id=1955.
    """

    super(FMRI_AA, self).__init__(root_dir, search_path, scan_type="nii",
      allow_multiple_files=allow_multiple_files, transforms=transforms,
      classes=classes)

  def _get_label_map(self, labels=["Autism", "Nonspectrum"]):
    """Load mapping between labels of all desired subjects and their
    corresponding class labels."""

    df = pd.read_csv(f"{self.root_dir}/../ndar_subject01.txt", sep="\t")

    # Keys are subject ids, values are phenotype labels
    label_map = {}
    for i in range(len(df)):
      row = df.iloc[i]
      sid = row["subjectkey"]
      pt = row["phenotype_description"]

      if sid not in label_map.keys() and pt in labels:
        label_map[sid] = pt

    return label_map
    

def main():
  """:meta private:"""
  dataset = FMRI_AA("../fmri_dataset_adhd_autism/imagingcollection01",
    r'sub-.*/ses-.*/fmap/.*\.nii\.gz', allow_multiple_files=False)

if __name__ == "__main__":
  main()