from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ada_mot import AdaMotTrainer
from .mot import MotTrainer

train_factory = {
    'mot': MotTrainer,
    'ada_mot': AdaMotTrainer
}
