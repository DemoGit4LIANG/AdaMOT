from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.lib.trains.ada_loss import AdaMotLoss
from src.lib.trains.base_trainer import BaseTrainer


class AdaMotTrainer(BaseTrainer):
    def __init__(self, opt, matcher_class, model, optimizer=None):
        self.matcher_class = matcher_class
        super(AdaMotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'cls_loss', 'l1_loss', 'giou_loss', 'id_loss']
        loss_fn = AdaMotLoss(opt, matcher=self.matcher_class(opt))
        return loss_states, loss_fn

    def save_result(self, output, batch, results):
        pass


# class AdaMotTrainer(BaseTrainer):
#     def __init__(self, opt, matcher, model, optimizer=None):
#         self.matcher = matcher
#         super(AdaMotTrainer, self).__init__(opt, model, optimizer=optimizer)
#
#     def _get_losses(self, opt):
#         loss_states = ['loss', 'cls_loss', 'l1_loss', 'giou_loss', 'id_loss']
#         loss_fn = AdaMotLoss(opt, matcher=self.matcher)
#         return loss_states, loss_fn
#
#     def save_result(self, output, batch, results):
#         pass
