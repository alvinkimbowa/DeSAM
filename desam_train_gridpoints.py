import os
import torch
from desam import sam_model_registry
from desam.training import run_training, run_testing
import argparse
import monai
from torch.utils.data import DataLoader
from utils.utils import Logger, split_prostatedataset
from utils.datasets import ProstateDataset
import torch.cuda.amp
from utils.dice_calculate import dice_calculate
from compile_results import compile


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--center', type=int, default=3)
parser.add_argument('--test_center', type=int, default=-1)
parser.add_argument('--model_type', type=str, default='vit_h')
parser.add_argument('--work_dir', type=str, default='E:/DeSAMData')

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--neg_points', type=int, default=1)
parser.add_argument('--grid', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--iou_thresh', type=float, default=0.5)

parser.add_argument('--mse', type=bool, default=False)
parser.add_argument('--sgd', type=bool, default=False)
parser.add_argument('--pred_embedding', type=bool, default=False)
parser.add_argument('--test_only', type=bool, default=False)
parser.add_argument('--eval_only', type=bool, default=False)
parser.add_argument('--random_validation', type=bool, default=False)
parser.add_argument('--mixprecision', type=bool, default=False)

'''
GPU info
with mix precision:
bs=1 | bs=2 | bs=4 | bs=8 | bs=16
4.1GB | 4.8GB | 5.7GB | 7.8GB | 11.6GB

without mix precision:
bs=8 10.9GB
'''

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

# prepare data path
work_dir = args.work_dir
os.makedirs(work_dir, exist_ok=True)
data_root = os.path.join(work_dir, 'image_embeddings', 'npz_files_{}').format(args.model_type)
if not os.path.exists(data_root):
    print('no precomputed image embeddings! please run precompute_embeddings.py first.')
    raise
task_name = 'desam_prostate_gridpoints_center{}'.format(args.center)
patientid_path = os.path.join(work_dir, 'raw_data', 'prostate_patientid.csv')
model_save_path = os.path.join(work_dir, 'results_folder', task_name)
os.makedirs(model_save_path, exist_ok=True)
logger = Logger(output_folder=model_save_path)

logger.print_to_log_file('gpuid: {}'.format(args.gpuid))
logger.print_to_log_file('center: {}'.format(args.center))
logger.print_to_log_file('test_center: {}'.format(args.test_center))
logger.print_to_log_file('work_dir: {}'.format(work_dir))
logger.print_to_log_file('pred_embedding: {}'.format(args.pred_embedding))
logger.print_to_log_file('mixprecision: {}'.format(args.mixprecision))

# prepare SAM model
model_zoo = {
    'vit_h': os.path.join(work_dir, 'checkpoint/sam_vit_h_4b8939.pth'),
    'vit_l': os.path.join(work_dir, 'checkpoint/sam_vit_l_0b3195.pth'),
    'vit_b': os.path.join(work_dir, 'checkpoint/sam_vit_b_01ec64.pth'),
}
device = 'cuda'
checkpoint = model_zoo[args.model_type]
desam = sam_model_registry[args.model_type](checkpoint=checkpoint)
desam.to(device)

# dataset split
train_patientid, val_patientid, test_patientid = split_prostatedataset(
    data_root=data_root,
    patientid_path=patientid_path,
    center=args.center,
    test_center=args.test_center,
    seed=args.center,
    random_validation=args.random_validation
)

# Set up the optimizer, hyperparameter tuning will improve performance here
if args.sgd:
    optimizer = torch.optim.SGD(desam.mask_decoder.parameters(), lr=args.lr, weight_decay=0, momentum=0.99)
else:
    optimizer = torch.optim.Adam(desam.mask_decoder.parameters(), lr=args.lr, weight_decay=0)

if args.mse:
    iou_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
else:
    iou_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    
dicece_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
scaler = torch.cuda.amp.GradScaler(enabled=args.mixprecision)

# prepare dataloader
train_dataset = ProstateDataset(data_root, train_patientid, args.neg_points)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = ProstateDataset(data_root, val_patientid, args.neg_points)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
desam.train()

# train, validation and test
test_center = "all" if args.test_center == -1 else args.test_center
if args.eval_only:
    if test_center == "all":
        logger.print_to_log_file('Running generalization evaluation on all centers except center {}'.format(args.center))
    else:
        logger.print_to_log_file('Running generalization evaluation on center {}'.format(test_center))
    save_mask = os.path.join(model_save_path, 'save_mask')
    dice, hd = dice_calculate(data_root, save_mask, test_patientid, edge_set_zero=50)
    logger.print_to_log_file('pred_iou_thresh: {}, mean_dice: {:.2f} +/- {:.2f}, mean_hd: {:.2f} +/- {:.2f}'.format(args.iou_thresh, dice[0], dice[1], hd[0], hd[1]))
    compile(dice, hd, args.center, args.test_center, logger)
elif args.test_only:
    if not os.path.exists(os.path.join(model_save_path, 'desam_model_best.pth')):
        print('use test_only after training!')
        raise
    
    if test_center == "all":
        logger.print_to_log_file('Running generalization evaluation on all centers except center {}'.format(args.center))
    else:
        logger.print_to_log_file('Running generalization evaluation on center {}'.format(test_center))
    desam = sam_model_registry[args.model_type](checkpoint=os.path.join(model_save_path, 'desam_model_best.pth'))
    desam.to(device=device)
    run_testing(
        model=desam, data_root=data_root, model_save_path=model_save_path, 
        iou_thresh=args.iou_thresh, ood_patientid=test_patientid, grid=args.grid, 
        logger=logger, pred_embedding=args.pred_embedding, device=device
    )
else:
    run_training(
        model=desam, max_num_epochs=args.epoch, logger=logger, model_save_path=model_save_path,
        optimizer=optimizer, initial_lr=args.lr, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, scaler=scaler, dicece_loss=dicece_loss,
        iou_loss=iou_loss, device=device, is_mixprecision=args.mixprecision
    )

    if test_center == "all":
        logger.print_to_log_file('Running generalization evaluation on all centers except center {}'.format(args.center))
    else:
        logger.print_to_log_file('Running generalization evaluation on center {}'.format(test_center))
    desam = sam_model_registry[args.model_type](checkpoint=os.path.join(model_save_path, 'desam_model_best.pth'))
    desam.to(device=device)
    run_testing(
        model=desam, data_root=data_root, model_save_path=model_save_path, 
        iou_thresh=args.iou_thresh, ood_patientid=test_patientid, grid=args.grid, 
        logger=logger, pred_embedding=args.pred_embedding, device=device
    )