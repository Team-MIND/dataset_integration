"""Define a parent BIDSDataset class for integration of BIDS data with 
PyTorch."""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import re
import os
import warnings
import nibabel as nib

################################################################################
# General Dataset
################################################################################


class TorchBIDS(Dataset):

  def __init__(self, root_dir, search_path, id_to_labels=None, scan_type="nii",
    allow_multiple_files=False, transforms=None, classes=None):
    """Generalized torch dataset to use with neuroimaging data in BIDS format.

    Subclass datasets can inherit from this class if they overwrite the
    _get_label_map() method.

    Args:
      root_dir: Path to root directory of BIDS dataset (directory containing
        all "sub-*" subdirs).
      search_path: Regex specifying which files of interest to match for each
        subject. E.g., if looking for all .nii.gz files within all "func"
        subdirectories for a subject, the regex would look something like
        r'ses-*./func/*.\.nii\.gz'. This regex should be passed as either a
        string or an r-string and should match files in each subdirectory.

    Note: The dataset will only track subjects who have folders in the root
    directory and whose folders contain relevant files matched by the passed
    regex search_path. If either one of these conditions is not met, the subject
    will not be included in the dataset.

    Keyword Args:
      id_to_labels: Dict-like, map subject ids (sub-*) to diagnosis labels.
        Subject labels are case sensitive!
      scan_type: {"nii"}, type of imaging files this dataset is required to
        open (default "nii").
      allow_multiple_files: If True, will allow the mapping id_to_files to
        contain a list of multiple files. This enables a single subject to be
        entered in the dataset ultiple times; once for each associated file.
        However, it may cause ambiguity when querying the dataset for the scan
        of a particular subject, so is set to False by default.
      transforms: Either a single tensor transformation or a list of them. Lists
        of transforms will be composed using torchvision.transform.Compose.
      classes: List of desired classes. If not None, subjects whose class labels
        are not in this set are ignored.
    """

    super(TorchBIDS, self).__init__()

    self.root_dir = root_dir

    # Add subject_id->label mapping as attribute
    self.id_to_labels = id_to_labels
    # Overwrite in case of subclass implementing alternative method
    self.id_to_labels = self._get_label_map()
    assert self.id_to_labels is not None, (
      "ID->Label mapping must be provided as an argument during "
      "TorchBIDS instaniation, or _get_label_map() must be overridden "
      "by subclass!"
    )

    # Determine list of distinct classes if list of predefined classes is not
    # passed
    if classes:
      # Filter id->label mapping
      self.id_to_labels = {k:self.id_to_labels[k] for k in 
        self.id_to_labels.keys() if self.id_to_labels[k] in classes}
      self.classes = classes
    else:
      # Extract all unique classes in id->label mapping
      self.classes = list(set([
        self.id_to_labels[k] for k in self.id_to_labels.keys()]))

    self.scan_type = scan_type
    self.allow_multiple_files = allow_multiple_files

    # Create mapping from subject id to all desired files
    self.id_to_files = {}

    # Search for appropriate files for each subject id in root directory
    for subject_id in self.id_to_labels.keys():
      # Search for appropriate directory
      for subdir in os.listdir(root_dir):
        path = f'{root_dir}/{subdir}'
        # If subject directory is found
        if os.path.isdir(path) and subject_id in subdir:
          # Traverse it for files
          for subdir, _, files in os.walk(path):
            # Combine subdir with regex for desired file path
            joined_re = re.compile("".join([subdir, "/", search_path]))
            for f in files:
              # If a file matching the specified regex is found, add it to
              # list of relevant files for this particular subject
              fpath = f"{subdir}/{f}"
              if re.match(joined_re, fpath):
                # Add file to list of relevant files for this subject,
                # or raise error if disallowing multiple entries per subject
                if (subject_id in self.id_to_files.keys() and not 
                  allow_multiple_files):
                  warnings.warn(RuntimeWarning((f"Multiple matching files found for "
                    f"subject ID {subject_id} with allow_matching_files = False:"
                    f"\n\t{self.id_to_files[subject_id]} (saved)"
                    f"\n\t{fpath} (discarded)\n")))
                elif subject_id in self.id_to_files.keys():
                  self.id_to_files[subject_id].append(fpath)
                elif not allow_multiple_files:
                  self.id_to_files[subject_id] = fpath
                else:
                  self.id_to_files[subject_id] = [fpath]
    
    # Store subject data in list
    self.data = []
    for s in self.id_to_files.keys():
      l = self.id_to_labels[s]
      fs = self.id_to_files[s]
      if allow_multiple_files:
        for f in fs:
          self.data.append((s, f, l))
      else:
        self.data.append((s, fs, l))

    # Compose transforms for loading scans
    self.transforms = transforms
    if isinstance(transforms, list):
      self.transforms = transforms.Compose(self.transforms)


  def __len__(self):
    """Return number of relevant files in this dataset.

    If allow_multiple_files is false, this will also be equal to the number of
    subjects.
    """

    if self.allow_multiple_files:
      total_files = []
      for _, fs, _ in self.data:
        total_files.extend(fs)
      return len(total_files)
    else:
      return len(self.data)


  def __getitem__(self, i):
    """Return the scan and label associated with the subject at index i."""

    sid, scan, label = self.data[i]
    scan = self.open_scan(scan)
    return (scan, label)

  
  def open_scan(self, scan):
    """Load scan data as a tensor depending on the type of scans to be read
    (currently only supports .nii files).
    """

    if self.scan_type == "nii":
      return self.open_nii(scan)
    else:
      raise ValueError(f"Unsupported scan type {self.scan_type}.")


  def open_nii(self, scan):
    """Open a scan given by a path to an nii file.

    Apply desired tensor transformations specified in dataset initialization.
    """

    # open thorugh nibabel use nib.load(pathname).get_data() to get array
    scan_data = torch.from_numpy(nib.load(scan).get_fdata())

    if self.transforms:
      return self.transforms(scan_data)
    return scan_data


  def _get_label_map(self):
    """Returns a dict whose keys are all desired subject ids and values are
    corresponding labels.
    """

    return self.id_to_labels



def main():
  # Nothing to see here
  pass
  

if __name__ == "__main__":
  main()