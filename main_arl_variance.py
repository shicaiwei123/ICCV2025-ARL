import argparse
import os
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.CramedDataset import CramedDataset,CramedDataset_swin
from dataset.KSDataset import KSDataset,KSDataset_swin
from dataset.VGGSoundDataset import VGGSound,VGGSound_swin
from dataset.AVEDataset import AVEDataset,AVEDataset_swin
from models.basic_model import AVClassifier_AUXI_Weight
from utils.utils import setup_seed, weight_init
from dataset.Kinect400 import Kinect400
import csv
import numpy as np
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film', 'share'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='./train_test_data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./train_test_data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70,150]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--gamma', type=float, default=1.0)

    parser.add_argument('--current_epoch', type=int, default=1)
    parser.add_argument('--T', type=float, default=8.0)
    parser.add_argument('--start', type=int, default=5)


    return parser.parse_args()



def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, acc_a, acc_v):
    criterion = nn.CrossEntropyLoss()


    scheduler.step()

    if epoch < 20:
        print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    model.module.args.current_epoch = epoch
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0

    for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))):

        # pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        out, out_a, out_v = model(spec.unsqueeze(1).float(),image.float())

        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss_f = criterion(out, label)


        # Modality analysis

        calculate_a = torch.mean(torch.abs(out_a), 0).sum().cpu().detach().numpy()
        calculate_b = torch.mean(torch.abs(out_v), 0).sum().cpu().detach().numpy()

        out_a = torch.softmax(out_a, dim=1)
        out_v = torch.softmax(out_v, dim=1)

        H_a = torch.sum(-out_a * torch.log(out_a + 1e-16), dim=1).mean().cpu().detach().numpy()
        H_v = torch.sum(-out_v * torch.log(out_v + 1e-16), dim=1).mean().cpu().detach().numpy()
        H_a = H_a.item()
        H_v = H_v.item()

        H_a_n, H_v_n = H_a / (H_a + H_v), H_v / (H_a + H_v)
        H_a = torch.clamp(torch.tensor(H_a_n), 0.3).cpu().detach().numpy()
        H_v = torch.clamp(torch.tensor(H_v_n), 0.3).cpu().detach().numpy()

        a_weight=1/H_a
        v_weight=1/H_v


        calculate_a_n, calculate_b_n = calculate_a / (calculate_a + calculate_b), calculate_b / (
                calculate_a + calculate_b)
        calculate_a, calculate_b = calculate_a_n, calculate_b_n


        # weight for ARL
        a_weight,v_weight=a_weight/(a_weight+v_weight),v_weight/(a_weight+v_weight)



        if step % 50 == 0:
            print("acc_a", acc_a, "Ha:", H_a, "accv:", acc_v, "hv:", H_v)
            print(calculate_a, calculate_b, a_weight, v_weight)


        if (acc_v + acc_a) != 0 and epoch > args.start:

            weight = F.softmax(
                torch.tensor([(calculate_b* a_weight) * args.T, calculate_a * v_weight * args.T]))
            weight_2 = F.softmax(torch.tensor([(calculate_b * a_weight) * 1, calculate_a * v_weight * 1]))
            if step % 50 == 0:
                print("log", [calculate_a, calculate_b, a_weight, v_weight],
                      weight.detach().cpu().numpy(), weight_2.detach().cpu().numpy())

            loss_cls = loss_f + (loss_a + loss_v) * args.gamma

        else:
            loss_cls = loss_f + (loss_a + loss_v) * 1
            model.module.args.audio_weight = 0.5
            model.module.args.visual_weight = 0.5


        loss = loss_cls
      
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)


        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
 

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader), _a_re / len(dataloader), _v_re / len(dataloader),


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'kinect400':
        n_classes = 400
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    # model.module.args.drop = 0
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]


        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)


            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    # model.module.args.drop = 1
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    # args.learning_rate=0.01
    # args.lr_decay_step='[30,60,90]'
    args.p = [1, 1]
    print(args)

    setup_seed(args.random_seed)
    if args.backbone=='swin':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    if args.backbone == 'resnet':
        model = AVClassifier_AUXI_Weight(args)
        model.apply(weight_init)
        raise EOFError

    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, eval(args.lr_decay_step), args.lr_decay_ratio)

    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = None
    else:
        raise ValueError


    if args.dataset == 'VGGSound':
        if args.backbone=='swin':
            train_dataset = VGGSound_swin(args, mode='train')
            test_dataset = VGGSound_swin(args, mode='test')
        else:
            train_dataset = VGGSound(args, mode='train')
            test_dataset = VGGSound(args, mode='test')

    elif args.dataset == 'KineticSound':
        if args.backbone=='swin':
            train_dataset = KSDataset_swin(args, mode='train')
            test_dataset = KSDataset_swin(args, mode='test')
        else:
            train_dataset = KSDataset(args, mode='train')
            test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'kinect400':
        train_dataset = Kinect400(args, mode='train')
        test_dataset = Kinect400(args, mode='test')
    elif args.dataset == 'CREMAD':
        if args.backbone=='swin':
            train_dataset = CramedDataset_swin(args, mode='train')
            test_dataset = CramedDataset_swin(args, mode='test')
        else:
            train_dataset = CramedDataset(args, mode='train')
            test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        if args.backbone == 'swin':
            train_dataset = AVEDataset_swin(args, mode='train')
            test_dataset = AVEDataset_swin(args, mode='test')
        else:
            train_dataset = AVEDataset(args, mode='train')
            test_dataset = AVEDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    log_path = os.path.join(args.ckpt_path, args.dataset + '_' + args.modality + '.csv')
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([1000, 1000, 1000])
    if args.train:

        best_acc = 0.0
        acc, acc_a, acc_v, acc_a_list, acc_v_list,H_a_sum,H_v_sum = 0, 0, 0, 0, 0,1,1

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))
            args.epoch_now = epoch

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                # print(acc_a_list,acc_v_list)
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args,
                                                                                                           epoch,
                                                                                                           model,
                                                                                                           device,
                                                                                                           train_dataloader,
                                                                                                           optimizer,
                                                                                                           acc, acc_a,
                                                                                                           acc_v,
                                                                                                           acc_a_list,
                                                                                                           acc_v_list,
                                                                                                           scheduler,
                                                                                                           )
                acc, acc_a, acc_v, acc_a_list, acc_v_list = valid(args, model, device, test_dataloader)
                print("acc", acc, acc_a, acc_v)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args, epoch,
                                                                                                           model,
                                                                                                           device,
                                                                                                           train_dataloader,
                                                                                                           optimizer,
                                                                                                           scheduler,
                                                                                                           acc_a, acc_v,
                                                                                                
                                                                                                           )
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                print("acc", acc, acc_a, acc_v)
                with open(log_path, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([acc, acc_a, acc_v])

            if acc > best_acc and epoch > 30:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_gamma_{}_T_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.gamma,
                                                          args.T,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                if scheduler is None:
                    saved_dict = {'saved_epoch': epoch,
                                  'modulation': args.modulation,
                                  'fusion': args.fusion_method,
                                  'acc': acc,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  }
                else:
                    saved_dict = {'saved_epoch': epoch,
                                  'modulation': args.modulation,
                                  'fusion': args.fusion_method,
                                  'acc': acc,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        # print(state_dict)
        model.load_state_dict(state_dict)
        # model.train()
        # model.eval()
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
