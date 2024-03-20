import torch
import numpy as np
from torchvision import transforms


class Spacer:
    def __init__(self, model):
        self.model = model

    def __call__(self, image: np.ndarray, replacer: np.ndarray) -> np.ndarray:
        # Make a segmentation mask
        mask_2d = self._make_segmentation(image)  # Shape: H x W x 1
        mask_3d = np.repeat(mask_2d, 3, axis=2)  # Shape: H x W x 3

        # Determine pieces that will replace people on an image
        people_replacer = replacer * mask_3d

        # Determine areas of an image to replace
        segmented_image = image * (~(mask_3d * image).astype(bool)).astype(int)

        # Combine segmented image and segmented background
        changed_image = people_replacer + segmented_image
        return changed_image

    def _make_segmentation(self, image: np.ndarray, threshold=0.5,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
        # Define transformations for an image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array(mean), np.array(std))
        ])

        # Transform image
        image = transform(image)

        # Make predictions with a UNet model
        logits = self.model(image[None, :, :, :])
        mask = torch.sigmoid(logits) > threshold
        mask = mask.squeeze(0)
        mask = mask.permute(1, 2, 0).numpy()
        return mask
