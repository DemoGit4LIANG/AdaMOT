from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar

from src.lib.models.data_parallel import DataParallel
from src.lib.utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch, epoch=1e5):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss_fn(outputs, batch, epoch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss_fn = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss_fn)
        self.optimizer.add_param_group({'params': self.loss_fn.parameters(), 'name': 'loss_net', 'lr': 1e-4})

        self.warm_up = opt.warm_up
        self.warmup_step = opt.warmup_step
        self.base_lr = {'backbone': 5e-5, 'cls_head': 5e-5, 'ltrb_head': 5e-5, 'id_head': 1e-4, 'loss_net': 1e-4}
        # self.base_lr = {'id_head': 1e-4, 'loss_net': 1e-4}

        if self.warm_up:
            print('====> warm step:', self.warmup_step)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            if self.warm_up:
                curr_step = len(data_loader) * (epoch - 1) + iter_id + 1
                if curr_step <= self.warmup_step:
                    set_wm_lr(self.optimizer, self.base_lr, curr_step=curr_step, wm_step=self.warmup_step)

            for k in batch:
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch, epoch)
            # TODO: Note here
            loss = loss.sum()

            num_obj = loss_stats['num_obj'].sum().item()
            loss /= num_obj

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].sum().item(), num_obj)
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    '''
        class MotTrainer(BaseTrainer):
            def __init__(self, opt, model, optimizer=None):
                super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)
        
            def _get_losses(self, opt):
                loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
                loss = MotLoss(opt)
                return loss_states, loss
    '''

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_wm_lr(optimizer, base_lr, curr_step, wm_step):
    for param_group in optimizer.param_groups:
        if param_group['name'] in base_lr:
            param_group['lr'] = base_lr[param_group['name']] * pow(curr_step / wm_step, 4)
        if curr_step == wm_step:
            print('===> wm finished:', param_group['name'], param_group['lr'])


if __name__ == '__main__':
    net = torch.nn.Linear(3, 10)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    print(optimizer.state_dict())
