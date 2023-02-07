from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.ada_jde import AdaJointDataset
from .dataset.jde import JointDataset


def get_dataset(dataset, task):
    if task == 'mot':
        return JointDataset
    elif task == 'ada_mot':
        return AdaJointDataset
    else:
        return None
