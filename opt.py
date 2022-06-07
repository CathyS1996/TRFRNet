import argparse
import os


parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/home/nono/SYT/data')
"-------------------dataset option--------------------------"
parse.add_argument('--A2B', type=str, default='_K2E')
# parse.add_argument('--dataset', type=str, default='kvasir_SEG')
# parse.add_argument('--train_data_dir', type=str, default='/home/nono/Dec/data/kvasir_SEG/Train')
# parse.add_argument('--valid_data_dir', type=str, default='/home/nono/Dec/data/kvasir_SEG/Valid')
parse.add_argument('--sdataset', type=str, default='kvasir_SEG')
parse.add_argument('--strain_data_dir', type=str, default='/home/nono/SYT/data/kvasir_SEG')
parse.add_argument('--tdataset', type=str, default='ETIS')
parse.add_argument('--ttrain_data_dir', type=str, default='/home/nono/SYT/data/ETIS/Train')
parse.add_argument('--tvalid_data_dir', type=str, default='/home/nono/SYT/data/ETIS/Valid')
parse.add_argument('--ttest_data_dir', type=str, default='/home/nono/SYT/data/ETIS/Test')

"-------------------training option-----------------------"
parse.add_argument('--expID', type=int, default=0)
parse.add_argument('--mode', type=str, default='test')
parse.add_argument('--load_ckpt', type=str, default='ck_best')
parse.add_argument('--model', type=str, default='TRFRNet')

parse.add_argument('--nEpoch', type=int, default=150)
parse.add_argument('--batch_size', type=float, default=4)
parse.add_argument('--iter_size', type=float, default=1)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--ckpt_period', type=int, default=10)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--lrD', type=float, default=1e-5)
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--power', type=float, default=0.9)

parse.add_argument('--nclasses', type=int, default=1)

opt = parse.parse_args()