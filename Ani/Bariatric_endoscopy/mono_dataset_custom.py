import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Custom dataset loader for monocular videos
class MonoDataset:
    def __init__(self, data_path, filenames, height, width, is_train=False, 
                 num_adjacent_frames=1):
        """
        Custom dataset loader for monocular videos.

        Args:
            data_path (str): Path to dataset root directory.
            filenames (list): List of image file paths (from train.txt or test.txt).
            height (int): Image height for resizing.
            width (int): Image width for resizing.
            is_train (bool): Whether dataset is used for training or evaluation.
            num_adjacent_frames (int): Number of adjacent frames to load for temporal data.
        """
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.num_adjacent_frames = num_adjacent_frames

        self.image_ext = '.png'  # Update this if images use a different format
        
        if self.is_train:
            # Training transformations with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((self.height, self.width), 
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flipping
                transforms.ColorJitter(
                    brightness=0.2,  # ±20%
                    contrast=0.2,    # ±20%
                    saturation=0.2,  # ±20%
                    hue=0.1          # ±10%
                ),  # Random color jitter with 50% chance
                transforms.ToTensor()
            ])
        else:
            # Validation transformations (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Load target image
        target_path = os.path.join(self.data_path, self.filenames[index])
        target_image = self._load_image(target_path)

        # Apply transformation to target image
        target_tensor = self.transform(target_image)

        # Load adjacent frames
        adjacent_images = []
        for offset in range(-self.num_adjacent_frames, self.num_adjacent_frames + 1):
            if offset == 0:  # Skip target frame itself
                continue
            adjacent_index = index + offset

            # Ensure adjacent index is within bounds
            if adjacent_index < 0 or adjacent_index >= len(self.filenames):
                adjacent_image = torch.zeros_like(target_tensor)  # Pad with a blank frame if out of bounds
            else:
                adjacent_path = os.path.join(self.data_path, self.filenames[adjacent_index])
                adjacent_image = self._load_image(adjacent_path)
                adjacent_image = self.transform(adjacent_image)

            adjacent_images.append(adjacent_image)

        # Stack adjacent frames along batch dimension
        adjacent_tensor = torch.stack(adjacent_images, dim=0)

        # Dummy intrinsics for simplicity (Replace with actual intrinsics if available)
        intrinsics = self._get_dummy_intrinsics()

        return {
            "image": target_tensor,                 # Target frame
            "adjacent_images": adjacent_tensor,     # Adjacent frames
            "path": self.filenames[index],          # Path of target frame
            "intrinsics": intrinsics                # Camera intrinsics
        }

    def _load_image(self, filepath):
        """Loads an image from a file."""
        try:
            with Image.open(filepath) as img:
                return img.convert('RGB')  # Convert to RGB
        except Exception as e:
            raise ValueError(f"Error loading image {filepath}: {e}")

    def _get_dummy_intrinsics(self):
        """Generates dummy camera intrinsics matrix."""
        focal_length = self.width / 2
        cx, cy = self.width / 2, self.height / 2
        intrinsics = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
