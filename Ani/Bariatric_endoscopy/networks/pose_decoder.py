# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=2):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # Initialize layer-specific "squeeze" convolutions
        self.convs = nn.ModuleDict({
            f"squeeze_{i}": nn.Conv2d(num_ch_enc[i], 256, 1)  # One "squeeze" layer per encoder level
            for i in range(len(num_ch_enc))
        })

        # Final pose prediction layer
        self.convs["pose"] = nn.Conv2d(256 * len(num_ch_enc), 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to reduce spatial dimensions

    def forward(self, last_features):
        """
        Args:
            last_features (list of tensors): Feature maps from pose_encoder.
        Returns:
            torch.Tensor: Predicted poses.
        """
        # Debug: Inspect input shapes
        # print("PoseDecoder input feature shapes:")
        # for i, f in enumerate(last_features):
        #     print(f"Feature {i} shape:", f.shape)

        # Target resolution: Match the spatial dimensions of the first feature map
        target_height, target_width = last_features[0].shape[-2:]

        # Apply "squeeze" convolution and upsample to target resolution
        cat_features = [
            F.interpolate(self.relu(self.convs[f"squeeze_{i}"](f)), size=(target_height, target_width), mode="bilinear", align_corners=False)
            for i, f in enumerate(last_features)
        ]

        # Concatenate features along the channel dimension
        cat_features = torch.cat(cat_features, dim=1)

        # Reduce spatial dimensions using global average pooling
        pooled_features = self.global_avg_pool(cat_features)  # Result: [batch_size, channels, 1, 1]

        # Predict poses using the concatenated and pooled features
        poses = self.convs["pose"](pooled_features)  # Result: [batch_size, 6 * num_frames_to_predict_for, 1, 1]

        # Flatten spatial dimensions and reshape to expected output format
        poses = poses.view(pooled_features.shape[0], self.num_frames_to_predict_for, 6)

        # Debug: Check output shape
        # print("Pose output shape:", poses.shape)

        return poses


# class PoseDecoder(nn.Module):
#     def __init__(self, num_ch_enc, num_input_features, 
#                  num_frames_to_predict_for=None, stride=1):
#         super(PoseDecoder, self).__init__()

#         self.num_ch_enc = num_ch_enc
#         self.num_input_features = num_input_features

#         if num_frames_to_predict_for is None:
#             num_frames_to_predict_for = num_input_features - 1
#         self.num_frames_to_predict_for = num_frames_to_predict_for

#         self.convs = OrderedDict()
#         self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
#         self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
#         self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
#         self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

#         self.relu = nn.ReLU()

#         self.net = nn.ModuleList(list(self.convs.values()))
    
#     def forward(self, input_features):
#         last_features = [f[-1] for f in input_features]

#         cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
#         cat_features = torch.cat(cat_features, 1)

#         out = cat_features
#         for i in range(3):
#             out = self.convs[("pose", i)](out)
#             if i != 2:
#                 out = self.relu(out)

#         out = out.mean(3).mean(2)

#         out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

#         axisangle = out[..., :3]
#         translation = out[..., 3:]

#         return axisangle, translation
