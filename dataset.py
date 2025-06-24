import torch
from torch.utils.data import Dataset, DataLoader
from azure.storage.blob import BlobServiceClient
import numpy as np
import random
from scipy import ndimage
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import os
from volumentations import *


class CTDataset(Dataset):
    def __init__(self, data_file, transform, prefix, rotate=False):
        self.dataset = data_file
        self.rotate = rotate
        self.blob_service_client = self._init_azure()

        self.prefix = prefix
        pos_weight = len(data_file) / (data_file["labels"] == 1).sum()
        neg_weight = len(data_file) / (data_file["labels"] == 0).sum()
        self.weights = {1: pos_weight, 0: neg_weight}
        # self.weights = torch.tensor([pos_weight, neg_weight]).cuda().detach()
        # self.weights /= self.weights.sum(0,keepdim=True)
        # self.transform = {
        #     torchio.CropOrPad((160,160,164)),
        #     # torchio.transforms.RandomFlip
        # }
        self.transform = transform
    
    def _init_azure(self):
        account_url = "https://eus2prdcornellaiechosa.blob.core.windows.net"
        default_credential = DefaultAzureCredential(managed_identity_client_id="b1d972cf-3885-46d8-8021-d553d2871d76")
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        return blob_service_client

    def __len__(self):
        return len(self.dataset)

    def rotate_func(self, volume):
        """Rotate the volume by a few degrees"""

        def scipy_rotate(volume):
            # define some rotation angles
            # angles = [-20, -10, -5, 5, 10, 20]
            angles = [-90]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < -500] = -500
            volume[volume > 500] = 500
            volume = 255 * (volume - volume.min()) / \
                (volume.max() - volume.min())
            # volume = (volume - volume.min()) / (volume.max() - volume.min())
            return volume

        augmented_volume = scipy_rotate(volume)
        return augmented_volume

    def __getitem__(self, idx):
        fname = self.dataset.iloc[idx]["Path_New"]
        blob_client = self.blob_service_client.get_blob_client(
            container="echo-data-lake", blob=fname
        )
        download_stream = blob_client.download_blob()
        download_arr = np.frombuffer(download_stream.readall(), dtype=np.float64)
        data = np.array(download_arr).reshape(164, 164, 164)
        data[data < -1000] = -1000
        data[data > 1000] = 1000
        # data = np.load(os.path.join(self.prefix, fname.split("/")[-1]))

        #!
        # if self.rotate:
        # data = self.rotate_func(data)
        data = self.transform(**{"image": data})["image"]
        # data = (data - -211.0977) / 525.811
        data = torch.FloatTensor(data)
        label = torch.tensor(self.dataset.iloc[idx]["labels"]).float()
        data = data.permute((2, 0, 1))

        #! repeat first frame -> (165,164,164)
        # data = torch.cat([data[0].unsqueeze(0), data])
        data = data.unsqueeze(0)

        return data, label, idx, self.weights[label.item()], self.dataset.iloc[idx]['ECHO_lvef_value'], self.dataset.iloc[idx]['CT_AccessionNumber']
