import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = True) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        # weighted_activations = weights[:, :, 12, :, :] * activations[:, :, 12, :, :]
        weighted_activations = weights * activations
        # print("weight", weights[:, :, None, None].shape,activations.shape,weighted_activations.shape)
        # print("weight", weights[:, :, 12, :, :].shape,activations[:, :, 12, :, :].shape,weighted_activations.shape)
        # (1, 48, 1, 1, 16, 1, 1) (1, 48, 16, 80, 80) (1, 48, 1, 48, 16, 80, 80)

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=2)
        return cam
        return cam.reshape(cam.shape[0],9,9)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = True) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = F.binary_cross_entropy_with_logits(outputs,targets.unsqueeze(1))
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer), outputs

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        slice, width, height = input_tensor.size(-3), input_tensor.size(-1), input_tensor.size(-2)
        return slice, width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # print(activations_list[0].shape,grads_list[0].shape)
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        cam_focus = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            if i == 0:
                #temporal
                # cam = np.maximum(cam, 0)
                # cam = cam.reshape(cam.shape[0], 9, 9)
                tmp = cam.mean(1)
                # tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min()) + 0.0001
                cam_focus.append(tmp)
                # scaled = scale_cam_image(cam, target_size)
                # cam_per_target_layer.append(scaled[None, :])
            else:
                cam = np.maximum(cam, 0)
                cam = np.transpose(cam, (1,0))
                cam = cam.reshape(cam.shape[0], 9, 9)
                cam = cam * cam_focus[0][:, None, None]
                scaled = scale_cam_image(cam, target_size)
                cam_per_target_layer.append(scaled[None, :])



        #     if i == 1:
        #         cam = np.maximum(cam,0)
        #     # cam = np.repeat(cam, 2,axis=0)
        #     if i== 1:
        #         cam = np.transpose(cam, (1,0))

        #     cam = cam.reshape(cam.shape[0], 9, 9)
        #     cam_focus.append(cam)
        #     scaled = scale_cam_image(cam, target_size)
        #     cam_per_target_layer.append(scaled[None, :])
        # import ipdb; ipdb.set_trace()
        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=0)
        # return scale_cam_image(cam_per_target_layer[1])
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        cam_per_target_layer = np.mean(cam_per_target_layer, axis=0)
        # cam_per_target_layer = 0.3 * cam_per_target_layer[0] + 0.7 * cam_per_target_layer[1]
        result = cam_per_target_layer
        # result = np.mean(cam_per_target_layer, axis=0)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
