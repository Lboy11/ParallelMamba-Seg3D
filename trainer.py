""" Retrieved from : https://github.com/gexing/SegMamba
"""
import itertools
import os
import numpy as np
import torch
from tqdm import tqdm
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.cuda.amp import GradScaler
from segmamba import SegMamba
from data import make_data_loader


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cuda_device = 0

        self.saver = None
        if self.cuda_device == 0:
            self.saver = Saver(args)
            self.saver.save_experiment_config()

        self.summary = None
        self.writer = None
        if self.cuda_device == 0:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.nchannels = make_data_loader(args)
        self._init_model(self.nchannels, self.nclass)
        self.evaluator = Evaluator(loss=args.loss.name, metrics=args.metrics)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.optim.lr,
                                      args.epochs, len(self.train_loader))
        self.resume()
        self.best_pred = 0
        self.scaler = GradScaler()
        self.global_step = 0


    def _init_model(self, nchannel, nclass):
        self.model = SegMamba(in_chans=1, out_chans=4, depths=[2, 2, 2, 2], feat_size=[32, 64, 128, 256])

        self.model = self.model.cuda()
        gpu_ids = '0,1'
        gpu_ids_list = list(map(int, gpu_ids.split(',')))
        self.model = torch.nn.parallel.DataParallel(self.model, device_ids=gpu_ids_list)
        train_params = [
            {'params': self.model.parameters(), 'lr': self.args.optim.lr}]

        # 定义优化器
        if self.args.optim.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                train_params, momentum=self.args.optim.momentum, weight_decay=self.args.optim.weight_decay)
        elif self.args.optim.name == 'adam':
            self.optimizer = torch.optim.Adam(
                train_params, weight_decay=self.args.optim.weight_decay)


    def resume(self, ):
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(
                self.args.resume, map_location=f'cuda:{self.cuda_device}')
            if self.args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
            self.args.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict = checkpoint['optimizer']

    def training(self, epoch):
        self.model.train()

        train_loss = 0.0

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, target_original = sample['image'], sample['label'], sample['original']
            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()
                target_original = target_original.cuda()

            self.scheduler(self.optimizer, epoch, i, self.best_pred)

            output, loss = self.forward_batch(image, target, target_original)

            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            if self.cuda_device == 0:
                self.writer.add_scalar(
                    'train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        if self.cuda_device == 0:
            print('[Epoch: {}]'.format(epoch))
            print('Loss: {:.3f}'.format(train_loss))
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        return train_loss / len(self.train_loader.dataset)


    def test(self, drop=None, epoch=0):
        dices = []
        dices_class = {}

        for l in reversed(range(self.nchannels)):
            for subset in itertools.combinations(list(range(self.nchannels)), l):
                dice, dice_class = self._test(drop=subset, epoch=epoch)

                dices.append(dice)
                dices_class[str(subset)] = dice_class

        if self.cuda_device == 0:
            self.save(np.mean(dices), dices_class, epoch)


    def _test(self, drop=[], epoch=0):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()

            for d in drop:
                image[:, d] = 0

            with torch.no_grad():
                output = self.predict(image, channel=drop)

            pred = output
            self.evaluator.add_batch(target.cpu().numpy(), pred.data.cpu().numpy())

        drop = str(drop).replace(',', '_').replace(' ', '')
        if drop.startswith('('):
            drop = drop[1:]
        if drop.endswith(')'):
            drop = drop[:-1]

        dice = self.evaluator.Dice_score()
        dice_class = self.evaluator.Dice_score_class()


        if self.cuda_device == 0:
            self.writer.add_scalar(f'test/dice_drop{drop}', dice, epoch)
            self.writer.add_scalar(
                f'test/dice_WT_drop{drop}', dice_class[0], epoch)
            self.writer.add_scalar(
                f'test/dice_TC_drop{drop}', dice_class[1], epoch)
            self.writer.add_scalar(
                f'test/dice_ET_drop{drop}', dice_class[2], epoch)

            # 打印当前测试模态下的Dice值和Dice类别
            print(f'Testing with modality {drop} dropped')
            print(f'Dice: {dice:.4f}')
            print(f'Dice: {dice_class}')

        return dice, dice_class



    def test1(self, drop=None, epoch=0):
        dices = []
        dices_class = {}
        for l in reversed(range(self.nchannels)):
            for subset in itertools.combinations(list(range(self.nchannels)), l):
                dice, dice_class = self._test1(drop=subset, l=l, epoch=epoch)

                dices.append(dice)
                dices_class[str(subset)] = dice_class
        if self.cuda_device == 0:
            self.save(np.mean(dices), dices_class, epoch)


    def _test1(self, drop=[], l=None, epoch=0):
        model = self.model.module

        checkpoint = torch.load('“Path to save your best generated model”')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        self.evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()

            for d in drop:
                image[:, d] = 0


            with torch.no_grad():
                output = self.predict(image, channel=drop)
                
            pred = output
            self.evaluator.add_batch(target.cpu().numpy(), pred.data.cpu().numpy())

        drop = str(drop).replace(',', '_').replace(' ', '')
        if drop.startswith('('):
            drop = drop[1:]
        if drop.endswith(')'):
            drop = drop[:-1]
        dice = self.evaluator.Dice_score()
        dice_class = self.evaluator.Dice_score_class()

        if self.cuda_device == 0:
            self.writer.add_scalar(f'test/dice_drop{drop}', dice, epoch)
            self.writer.add_scalar(
                f'test/dice_WT_drop{drop}', dice_class[0], epoch)
            self.writer.add_scalar(
                f'test/dice_TC_drop{drop}', dice_class[1], epoch)
            self.writer.add_scalar(
                f'test/dice_ET_drop{drop}', dice_class[2], epoch)

            print(f'Testing with modality {drop} dropped')
            print(f'Dice: {dice:.4f}')
            print(f'Dice: {dice_class}')

        return dice, dice_class


    def save(self, best_pred, dice_classes, epoch):
        new_pred = best_pred
        is_best = False

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

            with open(os.path.join(self.saver.experiment_dir, 'results.csv'), 'w') as f:
                for ii in dice_classes:
                    missing = str(ii).replace(',', '_').replace(' ', '')
                    f.writelines('{},{:.3f},{:.3f},{:.3f}\n'.format(missing, *[i * 100 for i in dice_classes[ii]]))


    def predict(self, image, channel=-1):
        output = self.model(x=image, channel=channel)

        return output



    def forward_batch(self, image, target, target_original):
        output, df, alpha_f = self.model(image)

        full_logits = torch.stack([df[l] for l in range(4)], dim=0)
        df_full = torch.mean(full_logits, dim=0)

        batch_y_original = target_original
        batch_y_ = batch_y_original.squeeze(1)
        batch_y_[batch_y_ == 4] = 3
        label_down4 = batch_y_.to(torch.int64)
        label_down4_onehot = torch.zeros(label_down4.size(0), 4, *label_down4.shape[1:], device=label_down4.device)
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        seg_loss = SegmentationLosses(self.args, nclass=self.nclass)

        loss = seg_loss.EnumerationLoss(output, target,  df, df_full, alpha_f, label_down4_onehot)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss








