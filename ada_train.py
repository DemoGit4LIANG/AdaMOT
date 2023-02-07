from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import json
import torch
import torch.utils.data
from torch import nn
from torchvision.transforms import transforms as T
from src.lib.datasets.dataset.ada_jde import collate_fn
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.logger import Logger
from src.lib.models.ada_model import AdaptiveNet
from src.lib.models.model import save_all_model
from src.lib.ada_opts import ada_opts
from src.lib.trains.ada_matcher import IdentityAwareLabelAssignment
from src.lib.trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    # opt.task
    # get_dataset -> JointDataset
    opt.task = 'ada_mot'
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    '''
        Dataset: JointDataset
    '''
    from src.lib.datasets.dataset.jde import JointDataset
    # dataset: JointDataset
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = ada_opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]) if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    '''
        # opt.arch: dla34
        # opt.head_conv: 256
        # opt.heads = {'hm': opt.num_classes, 1
                     'wh': 2 if not opt.ltrb else 4,
                     'id': opt.reid_dim}
    '''
    model = AdaptiveNet(backbone_name='dla', num_class=1, layers=34, head_conv=256, gn=False, is_train=True)
    start_epoch = 0

    init_lr = opt.lr
    print('init_lr:', init_lr)

    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        [{'params': model.backbone.parameters(), 'name': 'backbone'},
         {'params': model.cls_head.parameters(), 'name': 'cls_head'}, {'params': model.ltrb_head.parameters(), 'name': 'ltrb_head'},
         {'params': model.id_head.parameters(), 'lr': 1e-4, 'name': 'id_head'}], lr=init_lr, weight_decay=1e-4)

    # Get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    print('Starting training...')
    opt.task = 'ada_mot'
    Trainer = train_factory[opt.task]
    '''
        class AdaMotTrainer(BaseTrainer):
            def __init__(self, opt, matcher, model, optimizer=None):
    '''
    trainer = Trainer(opt, matcher_class=IdentityAwareLabelAssignment, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        if opt.resume:
            model, optimizer, start_epoch, trainer = _load_model(
                model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step, trainer=trainer)
        elif opt.id_aware:
            model, optimizer, start_epoch, trainer = _load_model(
                model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step, trainer=trainer, id_aware=opt.id_aware, id_net_path=opt.load_id_net)
        else:
            model, optimizer, start_epoch, _ = _load_model(
                model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    print('===> lr:', opt.lr)
    print('===> lr step:', opt.lr_step)

    if start_epoch in opt.lr_step:
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            print(old_lr)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_all_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                           epoch, model, optimizer, trainer=trainer)
        else:
            save_all_model(os.path.join(opt.save_dir, 'model_last.pth'),
                           epoch, model, optimizer, trainer=trainer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_all_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer, trainer=trainer)

            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.1
                param_group['lr'] = new_lr
                print('===> Drop LR to', new_lr)

        if epoch % 5 == 0 or epoch >= 25:
            save_all_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer, trainer=trainer)
    logger.close()


def _load_model(model, model_path, optimizer=None, resume=False,
                lr=None, lr_step=None, trainer=None, id_aware=False, id_net_path=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    # state_dict_ = checkpoint['state_dict']
    try:
        state_dict_ = checkpoint['model']
    except:
        state_dict_ = checkpoint['state_dict']

    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            # start_lr = lr
            lr_decay = 1.
            for step in lr_step:
                if start_epoch == step:
                    lr_decay *= 0.1
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * lr_decay
                param_group['lr'] = new_lr
                print('set optimizer lr from %s to %s' % (old_lr, new_lr))

            if trainer is not None:
                assert 'trainer' in checkpoint
                loss_fn_stat = checkpoint['trainer']
                trainer.loss_fn.load_state_dict(loss_fn_stat, strict=True)
                print('Resumed trainer successfully')
        else:
            print('No optimizer parameters in checkpoint.')

    if id_aware and id_net_path != '':
        checkpoint2 = torch.load(id_net_path, map_location=lambda storage, loc: storage)
        assert 'trainer' in checkpoint2
        loss_fn_stat = checkpoint2['trainer']
        trainer.loss_fn.load_state_dict(loss_fn_stat, strict=True)
        '''
            self.s_det = nn.Parameter(-1.85 * torch.ones(1))
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))
            self.id_aware = opt.id_aware
        '''
        trainer.s_det = nn.Parameter(-1.85 * torch.ones(1))
        trainer.s_id = nn.Parameter(-1.05 * torch.ones(1))
        trainer.id_aware = opt.id_aware
        print('Resume id fc successfully')
        print('Set trainer s_det: {}, s_id: {}'.format(trainer.s_det, trainer.s_id))
        print('Set trainer id_aware:', trainer.id_aware)

        return model, optimizer, start_epoch, trainer

    if optimizer is not None:
        return model, optimizer, start_epoch, trainer
    else:
        return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = ada_opts().parse()
    main(opt)
