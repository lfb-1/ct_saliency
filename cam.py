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
import matplotlib.pyplot as plt
import matplotlib.image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_val_dir", default="/home/fbl/Documents/Data/NYPCT", type=str
)
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
    df_train, transform, "/home/fbl/Documents/Data/NYPCT/ct_train/", rotate=False
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


cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    img = (img - img.min()) / (img.max() - img.min())
    img = cv2.merge((img, img, img))
    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def plot_cam(cam, dataloader):
    # count for how many class 0 and class 1 images to print
    maxcount0 = 30
    maxcount1 = 30
    count0 = 0
    count1 = 0

    for batch_idx, (input, label, idx, _) in enumerate(dataloader):
        input = input.cuda()
        label = label.cuda()
        # if count0 > maxcount0 and count1 > maxcount1:
        #     break

        # if label == 0:
        #     if count0 > maxcount0:
        #         continue
        #     else:
        #         count0 += 1
        # if label == 1:
        #     if count1 > maxcount1:
        #         continue
        #     else:
        #         count1 += 1

        grayscale_cam, output = cam(input, targets=label)
        grayscale_cam = grayscale_cam * 255
        if torch.logical_or(
            torch.logical_and(output.sigmoid() >= 0.5, label == 1),
            torch.logical_and(output.sigmoid() < 0.5, label == 0),
        ):
            continue

        if count0 > maxcount0 and count1 > maxcount1:
            break

        if label == 0:
            if count0 > maxcount0:
                continue
            else:
                count0 += 1
        if label == 1:
            if count1 > maxcount1:
                continue
            else:
                count1 += 1

        overlay = np.rot90(grayscale_cam, 3, axes=(1, 2))
        overlay = np.flip(overlay, axis=2) / 255
        overlay[overlay < 0.5] = 0
        overlay *= 255
        base_input = np.rot90(input.squeeze().detach().cpu().numpy(), 3, axes=(1, 2))
        base_input = np.flip(base_input, axis=2)

        # Set up the figure and axis
        fig, axes = plt.subplots(nrows=4, ncols=20, figsize=(31, 12))

        # Iterate over the images and axes
        for i, ax in enumerate(axes.flatten()):
            # ax.imshow(base_input[i * 2], cmap="gray")
            # ax.imshow(overlay[i], cmap="Reds", alpha=(overlay[i] / 256) ** 2)
            output = show_cam_on_image(base_input[i * 2], overlay[i], use_rgb=True)
            ax.imshow(output)
            ax.axis("off")  # Remove the axes ticks and labels

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=None, hspace=None)

        # Display the plot
        print(label)
        plt.savefig(
            f"cam_output/label{int(label)}/ct_gradcam_{idx.item()}.png",
            bbox_inches="tight",
        )


plot_cam(cam, train_dataloader)
