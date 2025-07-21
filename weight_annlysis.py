import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.KSDataset import KSDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.AVEDataset import AVEDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
import csv
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='./data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')

    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--beta', type=float, default=0)

    return parser.parse_args()


def main():
    args = get_arguments()
    args.pretrain=False
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    full_state_dict = torch.load(
        '/home/icml/shicaiwei/OGM-GE_CVPR2022/results/cramed/full_worker_32/best_model_of_dataset_CREMAD_Normal_alpha_0.8_optimizer_sgd_modulate_starts_0_ends_50_epoch_2_acc_0.40767045454545453.pth')[
        'model']

    model.load_state_dict(full_state_dict)
    fc_weight = model.module.fusion_module.fc_out.weight
    fc_weight = fc_weight.T

    fc_weight_mean = fc_weight[:,3]

    visual = torch.mean(torch.abs(fc_weight_mean[0:512]))
    print(torch.sum(visual))
    audio = torch.mean(torch.abs(fc_weight_mean[512:1024]))
    print(torch.sum(audio))
    print(1)


if __name__ == '__main__':
    main()
