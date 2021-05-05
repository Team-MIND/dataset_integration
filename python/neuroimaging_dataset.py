"""Define a parent dataset class for integration of BIDS/DICOM data with 
PyTorch."""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
import re
import os
import warnings

# For opening nii images
import nibabel as nib
# For opening DICOM images
import pydicom

################################################################################
# General Dataset
################################################################################


class TorchNI(Dataset):

  def __init__(self, root_dir, search_path, id_to_labels=None, scan_type="nii",
    allow_multiple_files=False, transforms=None, classes=None):
    """Generalized torch dataset to use with neuroimaging data in BIDS or
    DICOM format.

    Subclass datasets can inherit from this class if they overwrite the
    _get_label_map() method.

    Args:
      root_dir: Path to root directory of dataset.
      search_path: Function which takes as input a subject ID and outputs a
        regex specifying the absolute path (i.e., including the root directory) to all
        relevant files for this particular subject.

    Keyword Args:
      id_to_labels: Dict-like, map subject ids to diagnosis/phenotypic labels.
        Subject labels are case sensitive!
      scan_type: {"nii", "dicom"}, type of imaging files this dataset is
        required to open (default "nii").
      allow_multiple_files: If True, will allow the mapping id_to_files to
        contain a list of multiple files. This enables a single subject to be
        entered in the dataset multiple times; once for each associated file.
        However, it may cause ambiguity when querying the dataset for the scan
        of a particular subject, so is set to False by default.
      transforms: Either a single tensor transformation or a list of them. Lists
        of transforms will be composed using torchvision.transform.Compose.
      classes: List of desired classes. If not None, subjects whose class labels
        are not in this set are ignored.

    .. note:: 
      The dataset will only track subjects for which the dataset contains
      relevant files matched by the passed regex search_path.
    """

    super(TorchNI, self).__init__()

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
    for subject_id in list(self.id_to_labels.keys()):
      # Get path to desired files for this subject as regex
      path_to_files = search_path(subject_id)

      # Traverse root directory for files
      for dirpath, subdirs, files in os.walk(root_dir):
        for fpath in files:
          fpath = os.path.join(dirpath, fpath)
          # If a file matching the specified regex is found, add it to
          # list of relevant files for this particular subject
          if re.match(path_to_files, fpath):
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
<<<<<<< HEAD:python/neuroimaging_dataset.py
    elif self.scan_type == "dicom":
      return self.open_dicom(scan)
=======
    if self.scan_type == "dcm":
      return self.open_dcm(scan)
>>>>>>> 40fddef7ee7c52ca32d597bc60ff8c1e2651c9a9:neuroimaging_dataset.py
    else:
      raise ValueError(f"Unsupported scan type {self.scan_type}.")


  def open_nii(self, scan):
    """Open a scan given by a path to an nii file.

    Apply desired tensor transformations specified in dataset initialization.
    """

    # Open through nibabel use nib.load(pathname).get_data() to get array
    scan_data = torch.from_numpy(nib.load(scan).get_fdata())

    if self.transforms:
      return self.transforms(scan_data)
    return scan_data

  
  def open_dicom(self, scan):
    """Open a scan given by a path to a DICOM file.

    Apply desired tensor transformations specified in dataset initialization.
    """

    # TODO
    scan = pydicom.dcmread(scan)
    return scan


  def open_dcm(self, scan):
    """Open a scan given by a path to a dcm file.

    Apply desired tensor transformations specified in dataset initialization.
    """

    scan_data = torch.from_numpy(dcmread(scan).pixel_array)

    if self.transforms:
      return self.transforms(scan_data)
    return scan_data


  def _get_label_map(self):
    """Returns a dict whose keys are all desired subject ids and values are
    corresponding labels.
    """

    return self.id_to_labels



def main():
  """:meta private:"""

  # Test DICOM scan reading
  dataset = TorchNI("../fmri_dataset_adhd_autism/imagingcollection01",
    lambda x: r'.*', id_to_labels={"test":"Control"}, scan_type="dicom")
  

if __name__ == "__main__":
  main()