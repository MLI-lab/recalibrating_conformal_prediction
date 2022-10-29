from pathlib import Path
import numpy as np
import random
import json
from typing import Any, Callable, Iterable, Tuple, Union, Optional

# train_utils
from .train_utils import ImageNetEncodings

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url, check_integrity


dataset_mappings_path = "INSERT JSON PATH"
with open(dataset_mappings_path) as fn:
    DATASET_MAPPINGS = json.load(fn)

ACCEPTED_DATASET_TYPES = ['raw', 'encodings', 'logits']




def update_dataset_metadata(path: Union[str, Path], dataset_type: str, dataset: str = None, arch: str = None, is_no_extension: bool = False):
    # recursive dict update since we have level-2 dict
    import collections.abc
    def deep_update(source, overrides):
        """
        Update a nested dictionary or similar mapping.
        Modify ``source`` in place.
        """
        for key, value in overrides.items():
            if isinstance(value, collections.abc.Mapping) and value:
                returned = deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = overrides[key]
        return source


    # only consider supported dataset types
    assert dataset_type in ACCEPTED_DATASET_TYPES, f"Supported dataset types are {' '.join(ACCEPTED_DATASET_TYPES)}, but {dataset_type} was given!"

    path = Path(path)
    metadata_path = Path(dataset_mappings_path)
    
    # load current metadata
    with open(metadata_path) as fn:
        mappings_dict = json.load(fn)

    
    if dataset_type == 'raw':
        # if image dataset is given, check that it is a directory
        assert path.is_dir(), f"dataset_type is set to `raw`, but path {path} does not point to a directory!"

        if not dataset:
            # dataset name from folder if not given
            dataset = path.name

        update_dict = {dataset: {'raw': str(path)}}
        # mappings_dict[dataset].update({'raw': str(path)})
    else:
        assert arch, f"arch must be specified if dataset_type is set to {dataset_type}!"

        if path.is_dir():
            print(f"Path {path} points to a directory!\nEach file in this directory will be added to database with file name as the dataset name...")

            update_dict = {}
            for cur_path in path.iterdir():
                
                if is_no_extension:
                    update_dict.update({cur_path.name: {dataset_type: {arch: str(cur_path)}}})
                    # mappings_dict[cur_path.name][dataset_type].update({arch: str(cur_path)})
                else:
                    update_dict.update({cur_path.stem: {dataset_type: {arch: str(cur_path)}}})
                    # mappings_dict[cur_path.stem][dataset_type].update({arch: str(cur_path)})
        else:
            if not dataset:
                # dataset name from file if not given
                dataset = path.name if is_no_extension else path.stem

            update_dict = {dataset: {dataset_type: {arch: str(path)}}}
            # mappings_dict[dataset][dataset_type].update({arch: str(path)})

    
    # mappings_dict.update(update_dict)
    deep_update(mappings_dict, update_dict)

    # first copy the old metadata file for safety
    copy_path = metadata_path.parent / 'dataset_mappings_OUTDATED'
    metadata_path.rename(copy_path)

    with open(dataset_mappings_path, 'w') as fn:
        json.dump(mappings_dict, fn, indent=4)

    print(f"[DONE] Metadata file updated at {dataset_mappings_path}")






def subsample_dataset(dataset, num_samples_per_class=None, select_classes=[], num_classes=10):
    r"""
    args:
        dataset: Torchvision dataset that has the samples as tuples `(path, target)` in the field `dataset.samples` or as an array/tensor in `dataset.data` and targets in `dataset.targets`
        num_samples_per_class: Number of samples to use for each class (int)
        select_classes: List of classes to use (the rest will be discarded)
        num_classes: Number of total classes in the unaltered dataset
    returns:
        None (makes in-place changes to the dataset instead)
    """
    if not (num_samples_per_class or select_classes):
        print("No subsampling operation specified. Perhaps this function shouldn't have been called?")

    else:
        if not select_classes:
            select_classes = list(range(num_classes))

        sel_idx = []
        for lbl in select_classes:
            lbl_idx = [i for i, t in enumerate(dataset.targets) if t == lbl]
            sel_idx += random.sample(lbl_idx, (num_samples_per_class if num_samples_per_class else len(lbl_idx)))

        # subsample samples
        has_samples_attr = True
        try:
            new_samples = dataset.samples[sel_idx]
        except AttributeError:
            # dataset object does not have `samples` attribute. Assume that the samples are array/tensor in the `data` attribute.
            has_samples_attr = False
            new_samples = dataset.data[sel_idx]
        except TypeError:
            # assume a list
            new_samples = [dataset.samples[i] for i in sel_idx]
        finally:
            if has_samples_attr:
                dataset.samples = new_samples
            else:
                dataset.data = new_samples
        
        # subsample targets and fix the labels so that they go 0,1,2...
        try:
            new_targets = dataset.targets[sel_idx]
            for cur_idx, cur_cls in enumerate(select_classes):
                new_targets[new_targets==cur_cls] = cur_idx
        except TypeError:
            # assume a list
            new_lbl_dict = {cur_cls: cur_idx for cur_idx, cur_cls in enumerate(select_classes)}
            new_targets = [new_lbl_dict[dataset.targets[i]] for i in sel_idx]
        finally:
            dataset.targets = new_targets
        
        


def unlabel_dataset(dataset, num_labeled_samples=None, label_rate=None):
    r"""
    args:
        dataset: Torchvision dataset that return the samples with `dataset.samples()` and targets with `dataset.targets()`
        num_labeled_samples: Number of samples to retain labels
        label_rate: Portion of the dataset to retain labels (conflicts with `num_labeled_samples`)
    returns:
        None (makes in-place changes to the dataset instead)
    """

    assert (num_labeled_samples or label_rate) and not (num_labeled_samples and label_rate), "num_labeled_samples and percent_labeled cannot be missing or set at the same time!"

    if label_rate:
        assert (label_rate >= 0) and (label_rate <= 1), f"label_rate should be between [0,1], but {label_rate} was given"
        num_labeled_samples = int(len(dataset) * label_rate)

    sel_idx = random.sample(list(range(len(dataset))), len(dataset) - num_labeled_samples)

    dataset.targets[sel_idx] = -1




def get_class_target_mappings(datadir, folder_name_transform=lambda x: x):
    datadir = Path(datadir)
    classes_sorted = sorted([folder_name_transform(s.name) for s in datadir.glob('*') if s.is_dir()])
    return {k: i for i, k in enumerate(classes_sorted)}



def get_dataset(dataset: str, dataset_type: str = 'raw', arch: str = None, transform: Callable = None, limit_classes_by_dataset: Union[str,None] = None):

    assert dataset_type in ACCEPTED_DATASET_TYPES, f"dataset_type {dataset_type} is not supported. Please select one of the {ACCEPTED_DATASET_TYPES}"

    if not dataset_type == 'raw':
        assert arch is not None, f"[USAGE] Architecture needs to be given to use {dataset_type}!"

    try:
        cur_data_dict = DATASET_MAPPINGS[dataset.lower()]
        if dataset_type == 'raw':
            cur_dataset = cur_data_dict['raw']
        else:
            cur_dataset = cur_data_dict[dataset_type][arch]

    except KeyError as e:
        print(f"Encountered missing database entry for {dataset}-{dataset_type}-{arch}")
        raise e


    target_transform = None
    if not dataset_type == 'raw':
        assert transform is None, f"transform cannot be used for dataset of {dataset_type}"
    else:
        if transform is None:
            # transform for torchvision datasets
            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std= [0.229, 0.224, 0.225])
                        ])

        if dataset.lower() in ['imagenet-a', 'imagenet-r']:
            if dataset.lower() == 'imagenet-r':
                # No need to handle README files in imagenet-r directory as Pytorch checks is_dir for selecting the classes
                folder_name_transform = lambda x: x.split('_')[0]
            else:
                folder_name_transform = lambda x: x
            base_class_mappings = get_class_target_mappings(DATASET_MAPPINGS['imagenet-val']['raw'])
            new_class_mappings = get_class_target_mappings(cur_data_dict['raw'], folder_name_transform=folder_name_transform)

            new_target_mappings = {v:k for k, v in new_class_mappings.items()}

            target_transform = lambda x: base_class_mappings[new_target_mappings[x]]

    
    if not dataset_type == 'raw':
        dataset_object = ImageNetEncodings(Path(cur_dataset), train=False)
    else:
        dataset_object = datasets.ImageFolder(cur_dataset, transform, target_transform=target_transform)

    r""" ObjectNet only has 113 overlapping classes with ImageNet
        despite having 313 classes in total.
        Discard non-overlapping classes
    """
    if dataset.lower() == 'objectnet':
        from datasets.objectnet import extract_overlapping_classes

        extract_overlapping_classes(dataset_object)

    
    # only use a subset of the classes based on a provided dataset
    if limit_classes_by_dataset:
        if limit_classes_by_dataset.lower() == 'imagenet-r':
            # No need to handle README files in imagenet-r directory as Pytorch checks is_dir for selecting the classes
            folder_name_transform = lambda x: x.split('_')[0]
        else:
            folder_name_transform = lambda x: x

        base_class_mappings = get_class_target_mappings(DATASET_MAPPINGS['imagenet-val']['raw'])
        new_class_mappings = get_class_target_mappings(DATASET_MAPPINGS[limit_classes_by_dataset.lower()]['raw'], folder_name_transform=folder_name_transform)

        target_whitelist = [base_class_mappings[wnid] for wnid in new_class_mappings.keys()]

        filter_whitelisted_classes(dataset_object, target_whitelist)


    return dataset_object



def filter_whitelisted_classes(dataset: object, target_whitelist: Iterable):
    r"""Only keep whitelisted classes in the dataset"""
    sel_idx = []
    for lbl in target_whitelist:
        lbl_idx = [i for i, t in enumerate(dataset.targets) if t == lbl]
        sel_idx.extend(lbl_idx)

    # subsample samples
    try:
        new_samples = dataset.samples[sel_idx]
    except TypeError:
        new_samples = [dataset.samples[i] for i in sel_idx]
    finally:
        dataset.samples = new_samples

    # subsample targets
    try:
        new_targets = dataset.targets[sel_idx]
    except TypeError:
        new_targets = [dataset.targets[i] for i in sel_idx]
    finally:
        dataset.targets = new_targets

    

class CIFAR10V2(datasets.CIFAR10):

    dataset_remote_paths = {
        'v4': ['https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy', 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy'],
        'v6': ['https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy', 'https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy']
    }
    dataset_files = {
        'v4': ['cifar10.1_v4_data.npy', 'cifar10.1_v4_labels.npy'],
        'v6': ['cifar10.1_v6_data.npy', 'cifar10.1_v6_labels.npy']
    }

    def __init__(self, root: str, train: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, dataset_version: str = 'v6') -> None:
        if train:
            raise NotImplementedError("This object class does not support training sets.")
        super().__init__(root, train, transform, target_transform, download)

        self.dataset_version = dataset_version

        if download:
            self.download_cifar_v2()

        if not self._check_integrity_v2():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._load_v2_data()



    def _load_v2_data(self, load_tinyimage_indices: bool = False):
        r"""load CIFAR10.1 and overwrite existing CIFAR10 data and targets"""

        imagedata, labels = self.load_cifar_v2(self.root, self.dataset_version, load_tinyimage_indices)
        self.data = imagedata
        self.targets = labels.tolist()

    
    @staticmethod
    def load_cifar_v2(root: str, dataset_version: str = 'v6', load_tinyimage_indices: bool = False, verbose: bool = False) -> Tuple[np.ndarray, ...]:
        r"""load CIFAR10.1 and return data and labels"""

        assert dataset_version in ['v4', 'v6'], f"Unknown dataset version `{dataset_version}`"

        imagedata_fname, labelfname = CIFAR10V2.dataset_files[dataset_version]
        label_fpath = Path(root) / labelfname
        imagedata_fpath = Path(root) / imagedata_fname

        if verbose:
            print('Loading labels from file {}'.format(label_fpath))
        assert label_fpath.is_file()
        labels = np.load(label_fpath)

        if verbose:
            print('Loading image data from file {}'.format(imagedata_fpath))
        assert imagedata_fpath.is_file()
        imagedata = np.load(imagedata_fpath)

        assert len(labels.shape) == 1
        assert len(imagedata.shape) == 4
        assert labels.shape[0] == imagedata.shape[0]
        assert imagedata.shape[1] == 32
        assert imagedata.shape[2] == 32
        assert imagedata.shape[3] == 3

        if dataset_version == 'v6' or dataset_version == 'v7':
            assert labels.shape[0] == 2000
        elif dataset_version == 'v4':
            assert labels.shape[0] == 2021

        if not load_tinyimage_indices:
            return imagedata, labels
        else:
            raise NotImplementedError("Tiny image indices are not supported at this time.")
            # ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
            # ti_indices_filename = 'cifar10.1_' + dataset_version + '_ti_indices.json'
            # ti_indices_fpath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
            # print('Loading Tiny Image indices from file {}'.format(ti_indices_fpath))
            # assert pathlib.Path(ti_indices_fpath).is_file()
            # with open(ti_indices_fpath, 'r') as f:
            #     tinyimage_indices = json.load(f)
            # assert type(tinyimage_indices) is list
            # assert len(tinyimage_indices) == labels.shape[0]
            # return imagedata, labels, tinyimage_indices


    def _check_integrity_v2(self) -> bool:
        root = Path(self.root)
        for cur_file in self.dataset_files[self.dataset_version]:
            fpath = str(root / cur_file)
            if not check_integrity(fpath):
                return False
        return True
    
    def download_cifar_v2(self) -> None:
        r"""download dataset files from `https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/`"""

        if self._check_integrity_v2():
            print("Files already downloaded")
            return

        download_url_list = self.dataset_remote_paths[self.dataset_version]
        for cur_url in download_url_list:
            download_url(cur_url, self.root)



