# torch and visulization
import os.path

from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import load_dataset, load_param
from model.dataloader import TrainSetLoaderFits, TestSetLoaderFits
from model.smallTargetLoss import GeneralizedWassersteinDiceLoss

# model
from model.model_DNANet import Res_CBAM_block
from model.model_DNANet import DNANet

import re
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree

# random seed
seed = 1029
random.seed(1029)
os.environ['PYTHONHASHSEED'] = str(1029) # 为了禁止hash随机化，使得实验可复现
np.random.seed(1029)
torch.manual_seed(1029)
torch.cuda.manual_seed(1029)
torch.cuda.manual_seed_all(1029) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# ================= save model according recall =====================


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, self.val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor()])
        # input_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset = TrainSetLoaderFits(dataset_dir, img_id=train_img_ids, base_size=args.base_size,
                                      crop_size=args.crop_size, transform=input_transform, suffix=args.suffix,
                                      label_mode=args.label_mode)
        # testset         = TestSetLoaderFits (dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix,label_mode=args.label_mode)
        testset = TestSetLoaderFits(dataset_dir, img_id=self.val_img_ids, base_size=args.base_size,
                                    crop_size=args.crop_size, transform=input_transform, suffix=args.suffix,
                                    label_mode=args.label_mode)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)


        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'DNANet':
            model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model = model.cuda()
        ## Optimizing
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        ## Scheduling
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        ## Initialing weights
        if args.resume_from.strip() == '':
            model.apply(weights_init_xavier)
            self.start_epoch = args.start_epoch
        else:
            model.load_model(args.resume_from)
            try:
                checkpoint = torch.load(args.resume_from, map_location=lambda storage, loc: storage)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.start_epoch = checkpoint['epoch']
                del checkpoint
            except Exception as e:
                print(e)
                print('model file does not contains optimizer or scheduler state, cannot resume training')
        print("Model Initializing")
        self.model = model

        # Evaluation metrics
        self.best_f1 = 0
        self.test_fitsimg = fits.getdata("fitstrans/fitsdata/4/20200821152001999_7824_824122_LVT04.fits")
        M = np.array(
            [[0., 1.],
             [1., 0.]]
        )
        self.smallTargetLoss = GeneralizedWassersteinDiceLoss(dist_matrix=M, weighting_mode='GDL')

    # Training
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    # ============== 3 kinds losses ===============
                    ''' 1. iou-loss '''
                    loss += SoftIoULoss(pred, labels)
                    ''' 2. gfl-loss '''
                    loss += gfl_loss(pred, labels) / 800
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = 0
                # ============== 3 kinds losses ===============
                ''' 1. iou-loss '''
                loss = SoftIoULoss(pred, labels)
                ''' 2. gfl-loss '''
                loss += gfl_loss(pred, labels) / 800

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))

        self.train_loss = losses.avg
        save_train_loss = 'result/' + self.save_dir + '/' + self.save_prefix + '_training_loss.log'
        with open(save_train_loss, 'a') as f:
            f.write('Epoch:{}, loss:{}\n'.format(epoch, losses.avg))
        save_lr = 'result/' + self.save_dir + '/' + self.save_prefix + '_lr.log'
        with open(save_lr, 'a') as f:
            f.write('Epoch:{}, lr:{}\n'.format(epoch, self.scheduler.get_last_lr()))

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)

        self.model.eval()

        count = 0
        pre_sum = 0
        rec_sum = 0
        f1_sum = 0
        fa_sum = 0
        with torch.no_grad():
            for i, (data, data_name) in enumerate(tbar):
                data = data.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred = preds[-1]
                else:
                    pred = self.model(data)

                for num in range(len(pred)):
                    predsss = np.array((torch.sigmoid(pred[num]) > 0.4).cpu()).astype('int64') * 255
                    predsss = np.uint8(predsss)[0]

                    # 获取连通域
                    # ## not weighted
                    # pred_region = measure.regionprops_table(measure.label(predsss, connectivity=2),
                    #                                         properties=('label', 'centroid'))
                    # pred_region_array = np.concatenate(
                    #     [np.expand_dims(pred_region['centroid-0'], axis=1), np.expand_dims(pred_region['centroid-1'], axis=1),
                    #      np.expand_dims(pred_region['label'], axis=1)], axis=1)
                    ## weighted
                    test_fitsimg = fits.getdata(os.path.join('dataset', self.args.dataset, 'fits', '{}.fits'.format(data_name[num])))
                    pred_region = measure.regionprops_table(measure.label(predsss, connectivity=2),
                                                            properties=('label', 'centroid_weighted'),
                                                            intensity_image=test_fitsimg)
                    pred_region_array = np.concatenate(
                        [np.expand_dims(pred_region['centroid_weighted-0'], axis=1),
                         np.expand_dims(pred_region['centroid_weighted-1'], axis=1),
                         np.expand_dims(pred_region['label'], axis=1)], axis=1)
                    df_pred = pd.DataFrame(pred_region_array, columns=["row", "col", "label"])

                    # gt ipd 要改
                    fname = os.path.join('dataset', self.args.dataset, 'ipd', '{}.IPD'.format(data_name[num]))
                    df_gt = pd.read_table(fname, sep='\s+', header=None, encoding='utf-8',
                                          names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])
                    # calculate tp, fp, fn
                    thres = 1.5
                    all_pred = len(df_pred)
                    all_gt = len(df_gt)

                    ''' use KD-Tree '''
                    kd_pred = KDTree(df_pred[['col', 'row']].values)
                    match = kd_pred.query(df_gt[['col', 'row']].values, k=1)
                    tp = len(np.unique(match[1][match[0] < thres]))
                    fp = all_pred - tp
                    fn = all_gt - tp
                    if tp + fp == 0:
                        precision = 0
                    else:
                        precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision != 0 and recall != 0:
                        f1 = 2*precision*recall/(precision+recall)
                        faRate = (1 / precision - 1) * recall
                    else:
                        f1 = 0
                        faRate = 0
                    count += 1
                    pre_sum += precision
                    rec_sum += recall
                    f1_sum += f1
                    fa_sum += faRate
        precision = pre_sum / count
        recall = rec_sum / count
        f1 = f1_sum / count
        faRate = fa_sum / count
        print('Epoch:{}, precision:{}, recall:{}, f1:{}, faRate:{}'.format(epoch, precision, recall, f1, faRate))

        save_good_dir = 'result/' + self.save_dir + '/' + self.save_prefix + '_good_precision_recall.log'
        with open(save_good_dir, 'a') as f:
            f.write('Epoch:{}, precision:{}, recall:{}, f1:{}, faRate:{}\n'.format(epoch, precision, recall, f1, faRate))
        if precision > 0.45 and recall > 0.45 and f1 > self.best_f1:
            save_ckpt({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'recall': recall,
                'precision': precision
            }, save_path='result/' + self.save_dir,
                filename='recall_{}_precision_{}_'.format(recall, precision) + self.save_prefix + '_epoch{}'.format(
                    epoch) + '.pth.tar')
            self.best_f1 = f1


def main(args):
    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        trainer.scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    main(args)





