##
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from collections import defaultdict
import random
import os
from sklearn.model_selection import train_test_split
import copy
import wandb
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from utils import DiceLoss

##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

##
img_w = 224
img_h = 224


def random_crop(image, mask):
    random_h = random.randint(0, 500 - img_h - 1)  # 500-224-1
    random_w = random.randint(0, 500 - img_w - 1)  # 500-224-1
    image = image[random_h:random_h + img_h, random_w:random_w + img_w, :]
    mask = mask[random_h:random_h + img_h, random_w:random_w + img_w, :]
    return image, mask


def rotate(image, mask, angle):
    matrix_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    image = cv2.warpAffine(image, matrix_rotate, (img_w, img_h))
    mask = cv2.warpAffine(mask, matrix_rotate, (img_w, img_h))
    return image, mask


def blur(image):
    image = cv2.blur(image, (3, 3))
    return image


def add_noise(image):
    for i in range(200):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        image[temp_x][temp_y] = 255
    return image


def train_augmentation(image, mask):
    image, mask = random_crop(image, mask)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 90)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 180)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 270)
    if np.random.random() < 0.25:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if np.random.random() < 0.25:
        image = blur(image)

    if np.random.random() < 0.25:
        image = add_noise(image)
    return image, mask


def valid_augmentation(image, mask):
    image, mask = random_crop(image, mask)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 90)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 180)
    if np.random.random() < 0.25:
        image, mask = rotate(image, mask, 270)
    if np.random.random() < 0.25:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


##
class SatelliteDataset(Dataset):
    def __init__(self, image_filepaths, mask_filepaths, transform=None):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.mask_filepaths = mask_filepaths
        self.transform = transform
        self.ToTensor = ToTensorV2(p=1)
        self.normalization = A.Normalize(always_apply=True)

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        mask_filepath = self.mask_filepaths[idx]
        image = cv2.imread(image_filepath)
        mask = np.expand_dims(cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE), 2)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = self.normalization(image=image)["image"]
        image = self.ToTensor(image=image)["image"]
        mask = torch.Tensor(mask).reshape(img_h, img_w)
        return image, mask


##
class MetricMonitor:
    # Monitor metrics on interactive console
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return '|'.join([
            "{metric_name}:{avg:.{float_precision}f}".format(
                metric_name=metric_name,
                avg=metric['avg'],
                float_precision=self.float_precision)
            for (metric_name, metric) in self.metrics.items()
        ])


##
def pixel_accuracy(output, mask):
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    correct = torch.eq(output, mask).int().cpu()
    accuracy = float(correct.sum() / correct.numel())
    return accuracy


##
def mIoU(output, mask, n_classes, smooth=1e-6):
    output = torch.argmax(F.softmax(output, dim=1), dim=1).contiguous().view(-1)
    mask = mask.contiguous().view(-1)

    IoU_per_class = []
    for i in range(n_classes):
        true_class = output == i
        true_label = mask == i
        if true_label.long().sum().item() == 0:
            IoU_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
            IoU = (intersect + smooth) / (union + smooth)
            IoU_per_class.append(IoU)
    return np.nanmean(IoU_per_class)


##
def mdice_coef(output, mask, n_classes, smooth=1e-6):  # # 2TP/2TP+FP+FN
    output = torch.argmax(F.softmax(output, dim=1), dim=1).contiguous().view(-1)
    mask = mask.contiguous().view(-1)

    mdice_per_class = {}
    for i in range(n_classes):
        true_class = output == i
        true_label = mask == i
        if true_label.long().sum().item() == 0:
            mdice_per_class[i] = 0
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
            dice_coef = (2 * intersect + smooth) / (intersect + union + smooth)
            mdice_per_class[i] = dice_coef
    values = list(mdice_per_class.values())
    return mdice_per_class, np.sum(values) / np.count_nonzero(values)


##
def train(train_loader, model, optimizer, epoch):
    metric_monitor = MetricMonitor()
    train_loss = 0
    train_accuracy = 0
    train_IoU = 0
    train_dice_coef = 0
    train_dc_0 = 0
    train_dc_1 = 0
    train_dc_2 = 0
    train_dc_3 = 0
    train_dc_4 = 0
    train_dc_5 = 0
    train_dc_6 = 0
    train_dc_8 = 0
    train_dc_9 = 0
    model.train()
    stream = tqdm(train_loader)
    for (images, masks) in stream:
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        output = model(images)
        ce_loss = ce(output, masks)
        dice_loss = dl(output, masks, softmax=True)
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        acc = pixel_accuracy(output, masks)
        IoU = mIoU(output, masks, n_classes=10)
        dc_output = mdice_coef(output, masks, n_classes=10)
        dice_coef = dc_output[1]
        train_loss += loss.item()
        train_accuracy += acc
        train_IoU += IoU
        train_dice_coef += dice_coef
        train_dc_0 += dc_output[0][0]
        train_dc_1 += dc_output[0][1]
        train_dc_2 += dc_output[0][2]
        train_dc_3 += dc_output[0][3]
        train_dc_4 += dc_output[0][4]
        train_dc_5 += dc_output[0][5]
        train_dc_6 += dc_output[0][6]
        train_dc_8 += dc_output[0][8]
        train_dc_9 += dc_output[0][9]
        loss.backward()
        optimizer.step()
        metric_monitor.update('loss', loss.item())
        metric_monitor.update('accuracy', acc)
        metric_monitor.update('IoU', IoU)
        metric_monitor.update('dice_coef', dice_coef)
        stream.set_description(f"Epoch: {epoch}. Train: {metric_monitor}")
    wandb.log({
        "Train Loss": train_loss / len(train_loader),
        "Train Accuracy": train_accuracy / len(train_loader),
        "Train mean IoU": train_IoU / len(train_loader),
        "Train mean dice coefficient": train_dice_coef / len(train_loader),
        "Train dice coefficient for No Data": train_dc_0 / len(train_loader),
        "Train dice coefficient for Cultivated Land": train_dc_1 / len(train_loader),
        "Train dice coefficient for Forest": train_dc_2 / len(train_loader),
        "Train dice coefficient for Grassland": train_dc_3 / len(train_loader),
        "Train dice coefficient for Shrubland": train_dc_4 / len(train_loader),
        "Train dice coefficient for Water": train_dc_5 / len(train_loader),
        "Train dice coefficient for Wetlands": train_dc_6 / len(train_loader),
        "Train dice coefficient for Artificial Surface": train_dc_8 / len(train_loader),
        "Train dice coefficient for Bareland": train_dc_9 / len(train_loader),
    })


##
def valid(valid_loader, model, epoch):
    metric_monitor = MetricMonitor()
    valid_loss = 0
    valid_accuracy = 0
    valid_IoU = 0
    valid_dice_coef = 0
    valid_dc_0 = 0
    valid_dc_1 = 0
    valid_dc_2 = 0
    valid_dc_3 = 0
    valid_dc_4 = 0
    valid_dc_5 = 0
    valid_dc_6 = 0
    valid_dc_8 = 0
    valid_dc_9 = 0
    model.eval()
    stream = tqdm(valid_loader)
    for (images, masks) in stream:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        output = model(images)
        ce_loss = ce(output, masks)
        dice_loss = dl(output, masks)
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        acc = pixel_accuracy(output, masks)
        IoU = mIoU(output, masks, n_classes=10)
        dc_output = mdice_coef(output, masks, n_classes=10)
        dice_coef = dc_output[1]
        valid_loss += loss.item()
        valid_accuracy += acc
        valid_IoU += IoU
        valid_dice_coef += dice_coef
        valid_dc_0 += dc_output[0][0]
        valid_dc_1 += dc_output[0][1]
        valid_dc_2 += dc_output[0][2]
        valid_dc_3 += dc_output[0][3]
        valid_dc_4 += dc_output[0][4]
        valid_dc_5 += dc_output[0][5]
        valid_dc_6 += dc_output[0][6]
        valid_dc_8 += dc_output[0][8]
        valid_dc_9 += dc_output[0][9]
        metric_monitor.update('loss', loss.item())
        metric_monitor.update('accuracy', acc)
        metric_monitor.update('IoU', IoU)
        metric_monitor.update('dice_coef', dice_coef)
        stream.set_description(f"Epoch: {epoch}. Valid: {metric_monitor}")
    wandb.log({
        "Valid Loss": valid_loss / len(valid_loader),
        "Valid Accuracy": valid_accuracy / len(valid_loader),
        "Valid mean IoU": valid_IoU / len(valid_loader),
        "Valid mean dice coefficient": valid_dice_coef / len(valid_loader),
        "Valid dice coefficient for No Data": valid_dc_0 / len(valid_loader),
        "Valid dice coefficient for Cultivated Land": valid_dc_1 / len(valid_loader),
        "Valid dice coefficient for Forest": valid_dc_2 / len(valid_loader),
        "Valid dice coefficient for Grassland": valid_dc_3 / len(valid_loader),
        "Valid dice coefficient for Shrubland": valid_dc_4 / len(valid_loader),
        "Valid dice coefficient for Water": valid_dc_5 / len(valid_loader),
        "Valid dice coefficient for Wetlands": valid_dc_6 / len(valid_loader),
        "Valid dice coefficient for Artificial Surface": valid_dc_8 / len(valid_loader),
        "Valid dice coefficient for Bareland": valid_dc_9 / len(valid_loader),
    })


##
def predict(image):
    image = A.PadIfNeeded(p=1, min_width=672, min_height=672, border_mode=cv2.BORDER_CONSTANT)(image=image)["image"]
    image = A.Normalize(always_apply=True)(image=image)["image"]
    image = ToTensorV2(p=1)(image=image)["image"]

    def model_output(patches):
        output = model(patches)
        return torch.argmax(F.softmax(output, dim=1), dim=1)

    image = image.to(device)
    patch_0 = image[:, :img_h, :img_w].unsqueeze(0)
    patch_1 = image[:, :img_h, img_w:2 * img_w].unsqueeze(0)
    patch_2 = image[:, :img_h, 2 * img_w:].unsqueeze(0)
    patch_3 = image[:, img_h:2 * img_h, :img_w].unsqueeze(0)
    patch_4 = image[:, img_h:2 * img_h, img_w:2 * img_w].unsqueeze(0)
    patch_5 = image[:, img_h:2 * img_h, 2 * img_w:].unsqueeze(0)
    patch_6 = image[:, 2 * img_h:, :img_w].unsqueeze(0)
    patch_7 = image[:, 2 * img_h:, img_w:2 * img_w].unsqueeze(0)
    patch_8 = image[:, 2 * img_h:, 2 * img_w:].unsqueeze(0)
    predicted_patches = model_output(
        torch.cat((patch_0, patch_1, patch_2, patch_3, patch_4, patch_5, patch_6, patch_7, patch_8), dim=0))
    B, H, W = predicted_patches.shape
    predicted_image = predicted_patches.view(3, 3, H, W)
    predicted_image = predicted_image.permute(0, 2, 1, 3).contiguous().view(H * 3, W * 3)
    predicted_image = predicted_image.cpu().detach().numpy()
    return A.CenterCrop(height=500, width=500)(image=predicted_image)["image"]



##
if __name__ == "main":
    wnb_username = 'username'
    wnb_project_name = "project_name"
    wandb.init(entity=wnb_username, project=wnb_project_name)

    config = wandb.config
    config.batch_size = 2
    config.test_batch_size = 2
    config.epochs = 200
    config.lr = 0.05
    config.momentum = 0.9
    config.no_cuda = False
    config.seed = 42
    config.log_interval = 10
    config.weight_decay = 1e-5

    model = SwinTransformerSys(img_size=224, window_size=7, num_classes=10).to(device)
    pretrained_dict = torch.load("swin_tiny_patch4_window7_224.pth", map_location=device)['model']
    model_dict = model.state_dict()
    full_dict = copy.deepcopy(pretrained_dict)
    for k, v in pretrained_dict.items():
        if "layer." in k:
            current_layer_num = 3 - int(k[7:8])
            current_k = "layers_up." + str(current_layer_num) + k[8:]
            full_dict.update({current_k: v})
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape != model_dict[k].shape:
                del full_dict[k]

    model.load_state_dict(full_dict, strict=False)

    df = pd.read_csv("df.csv", index_col=0)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = SatelliteDataset(df_train["image"].values, df_train["mask"].values, transform=train_augmentation)
    valid_dataset = SatelliteDataset(df_valid["image"].values, df_valid["mask"].values, transform=valid_augmentation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=2)

    num_epochs = config.epochs
    ce = nn.CrossEntropyLoss().to(device)
    dl = DiceLoss(10).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=config.lr, momentum=config.momentum,weight_decay=config.weight_decay)
    wandb.watch(model, log="all")

    for epoch in range(1, num_epochs + 1):
        train(train_loader, model, optimizer, epoch)
        valid(valid_loader, model, epoch)
