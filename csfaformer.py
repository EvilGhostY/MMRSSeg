from torch.utils.data import DataLoader
from model.datasets.potsdam_dataset import *
from model.models.CSFAFormer import CSFAFormer
from model.losses.CSFAFormerLoss import CSFAFormerLoss
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import torch.nn as nn

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "CSFAFormer"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "CSFAFormer"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 2
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path =None
resume_ckpt_path = None
#  define the network
vis_channels = 2
ir_channels = 1
dsm_channels = 1
Dual_Branch = True

#  define the network
#net =GDGNet(num_classes=num_classes,vis_channels =vis_channels)
#net = EncoderDecoder()
net = CSFAFormer(num_classes=num_classes)
# define the loss
loss = CSFAFormerLoss(ignore_index=ignore_index)# , decode_channels=64
use_aux_loss = True


# define the dataloader
train_dataset = PotsdamDataset(data_root='MMRSSeg/data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(data_root='MMRSSeg/data/potsdam/test',transform=val_aug)
test_dataset = PotsdamDataset(data_root='MMRSSeg/data/potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
if Dual_Branch:
    layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay),
                       "dsmbackbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
else:
    layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)