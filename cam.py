import argparse
import numpy as np
from archs import CTViT_Encoder, extract_encoder
import pandas as pd
import os
import torch
from dataset import CTDataset
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from volumentations import CenterCrop, Compose, Flip
from captum.attr import GuidedGradCam
import matplotlib.pyplot as plt
import matplotlib.image

parser = argparse.ArgumentParser()
parser.add_argument("--train_val_dir", default="/home/fbl/Documents/Data/", type=str)
parser.add_argument("--weight_dir", default="/home/fbl/Documents/source/pretrains/")
parser.add_argument("--train_val_name", default=59, type=int)
parser.add_argument("--batch_size", default=1, type=int)
args = parser.parse_args()
df_train = pd.read_csv(
    os.path.join(
        args.train_val_dir, f"train_dataset_{args.train_val_name}_fixed_pixel.csv"
    )
)

# df_val = pd.read_csv(
#     os.path.join(
#         args.train_val_dir, f"val_dataset_{args.train_val_name}_fixed_pixel.csv"
#     )
# )
gradients = None
activations = None
transform = Compose([CenterCrop((160, 160, 164), always_apply=True, p=1)], p=1.0)

train_dataset = CTDataset(
    df_train, transform, "/home/fbl/Documents/Data/ct_train/", rotate=False
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=8,
)

device = torch.device("cuda")
model = CTViT_Encoder(
    dim=512,
    codebook_size=8192,
    image_size=160,
    patch_size=16,
    temporal_patch_size=2,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

model = model.to(device)
new_pt = extract_encoder(os.path.join(args.weight_dir, "ctvit_pretrained.pt"))
model.load_state_dict(new_pt)

model_best = torch.load(
    os.path.join(args.weight_dir, "model_best_0.7607340012073844.pth.tar")
)["model"]
model.load_state_dict(model_best)

print("Model and weight loaded")


def reshape_transform(tensor, height=7, width=7):
    result = tensor
    # result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # result = result.transpose(2, 3).transpose(1, 2)
    return result


model.eval()

target_layers = [model.enc_temporal_transformer.layers[-1][3][0]]

guided_gc = GuidedGradCam(model, model.enc_spatial_transformer.layers[-1][3][0])

cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)


def plot_cam(cam, dataloader):
    for batch_idx, (input, label, idx, _) in enumerate(dataloader):
        input = input.cuda()
        label = label.cuda()
        grayscale_cam = cam(input, targets=label) * 255
        input_norm = input.squeeze().detach().cpu().numpy()
        input_norm = (input_norm - input_norm.min()) / (input_norm.max() - input_norm.min())
        input_norm *= 255
        output = 0.3 * grayscale_cam + 0.7 * input_norm
        output = np.rot90(output,3, axes=(1,2))
        if label == 1:
            for i in range(164):
                matplotlib.image.imsave(f"cam_output/ct_gradcam_{idx.item()}_{i}.png", output[i])
        print(grayscale_cam.shape)


plot_cam(cam, train_dataloader)
