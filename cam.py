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
    "--train_val_dir", default="/mnt/azureml/cr/j/58dd9485ffae4bbd8d39316bb3ce9c93/cap/data-capability/wd/data_dir/bed_remove_raw/298_Preprocessed_CT_ECHO_columbia_test_new.csv", type=str
)
parser.add_argument(
    "--weight_dir", default="/mnt/azureml/cr/j/58dd9485ffae4bbd8d39316bb3ce9c93/cap/data-capability/wd/output_dir/bed_removal_final_40k/123_16_model.pth.tar")
parser.add_argument("--train_val_name", default=59, type=int)
parser.add_argument("--batch_size", default=1, type=int)
args = parser.parse_args()
# df_train = pd.read_csv(
#     os.path.join(
#         args.train_val_dir, f"train_dataset_{args.train_val_name}_fixed_pixel.csv"
#     )
# )

df_train = pd.read_csv(
    args.train_val_dir
)

# df_val = pd.read_csv(
#     os.path.join(
#         args.train_val_dir, f"val_dataset_{args.train_val_name}_fixed_pixel.csv"
#     )
# )
gradients = None
activations = None
transform = Compose([CenterCrop((144, 144, 164), always_apply=True, p=1)], p=1.0)

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
    image_size=144,
    patch_size=16,
    temporal_patch_size=2,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

model = model.to(device)
# new_pt = extract_encoder(os.path.join(args.weight_dir, "ctvit_pretrained.pt"))
# model.load_state_dict(new_pt)

model_best = torch.load(
    args.weight_dir
)["state_dict"]
model.load_state_dict(model_best)

print("Model and weight loaded")


def reshape_transform(tensor, height=7, width=7):
    result = tensor
    # result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # result = result.transpose(2, 3).transpose(1, 2)
    return result


model.eval()

target_layers = [model.enc_spatial_transformer.layers[-1][3][0], model.enc_temporal_transformer.layers[-1][3][0]]


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
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap * \
        (cv2.merge((mask, mask, mask)) > 0) + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def plot_cam(cam, dataloader):
    # count for how many class 0 and class 1 images to print
    maxcount0 = 50
    maxcount1 = 50
    count0 = 0
    count1 = 0

    for batch_idx, (input, label, idx, _, lvef) in enumerate(dataloader):
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
        if output.sigmoid() < 0.5 and label == 1:
            continue
        elif output.sigmoid() >= 0.5 and label == 0:
            continue
        elif torch.amax(input, dim=(1, 2, 3, 4)) <= 0:
            continue

        if count0 > maxcount0 and count1 > maxcount1:
            break

        if label == 0:
            continue
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
        # overlay = grayscale_cam
        orig_max = np.max(overlay)
        # overlay = overlay / orig_max
        overlay = np.flip(overlay, axis=2) / np.max(overlay)
        overlay = (overlay - 0.0) / (np.max(overlay) - 0.0)
        overlay[overlay < 0] = 0
        overlay *= orig_max
        base_input = np.rot90(
            input.squeeze().detach().cpu().numpy(), 3, axes=(1, 2))
        # base_input = input.squeeze().detach().cpu().numpy()
        base_input = np.flip(base_input, axis=2)

        # Set up the figure and axis
        fig, axes = plt.subplots(nrows=8, ncols=10, figsize=(20, 12))

        # Iterate over the images and axes
        for i, ax in enumerate(axes.flatten()):
            # ax.imshow(base_input[i * 2], cmap="gray")
            # ax.imshow(overlay[i], cmap="Reds", alpha=(overlay[i] / 256) ** 2)
            output = show_cam_on_image(
                base_input[i * 2], overlay[i], use_rgb=True)
            ax.imshow(output)
            ax.axis("off")  # Remove the axes ticks and labels

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=-0.7, hspace=0.1)
        # fig.tight_layout()

        # Display the plot
        print(label)
        plt.savefig(
            f"cam_output/label{int(label)}/ct_gradcam_{idx.item()}_LVEF{lvef.item()}.png",
            bbox_inches="tight",
        )
        # plt.savefig(
        #     f"/mnt/azureml/cr/j/9077c72b6a5f4295b4091f38eb00753d/cap/data-capability/wd/output_dir/bed_removal_final_40k/cam_output/label{int(label)}/ct_gradcam_{idx.item()}_LVEF{lvef.item()}.png", bbox_inches="tight"
        # )


plot_cam(cam, train_dataloader)
