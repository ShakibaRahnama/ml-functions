import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

### Loss function for monocular videos
# Calculates combined loss, including photometric reconstruction loss, smoothness loss, and auto-masking
def compute_depth_loss(inputs, adjacent_inputs, outputs, pose_outputs, intrinsics, 
                       scales=[0, 1, 2, 3]):
    """
    Compute Monodepth2-style depth loss for monocular videos.

    Parameters:
        inputs (torch.Tensor): Target frames (B, C, H, W).
        adjacent_inputs (torch.Tensor): Adjacent video frames (B, N, C, H, W) or (B, C, H, W).
        outputs (dict): Predicted depth maps and reconstructed images.
        pose_outputs (torch.Tensor): Predicted poses (B, N, 6).
        intrinsics (torch.Tensor): Camera intrinsics matrix (B, 3, 3).
        scales (list): List of scales to compute loss.

    Returns:
        torch.Tensor: Combined depth loss.
    """
    photometric_weight = 1.0
    smoothness_weight = 0.001
    total_loss = 0

    for scale in scales:
        # Predicted disparity and depth at current scale
        disp = outputs[("disp", scale)]  # (B, 1, H, W)
        depth = 1.0 / (disp + 1e-7)      # Convert disparity to depth

        # Resize inputs to match current scale
        target_scaled = F.interpolate(inputs, size=disp.shape[-2:], mode="bilinear", align_corners=False)

        # Handle 4D or 5D adjacent_inputs
        if adjacent_inputs.dim() == 5:  # (B, N, C, H, W)
            adjacent_scaled = F.interpolate(
                adjacent_inputs.reshape(-1, adjacent_inputs.shape[2], adjacent_inputs.shape[3], adjacent_inputs.shape[4]),
                size=disp.shape[-2:], mode="bilinear", align_corners=False
            )
            adjacent_scaled = adjacent_scaled.view(
                adjacent_inputs.shape[0], adjacent_inputs.shape[1], 
                *adjacent_scaled.shape[-3:]
            )
        elif adjacent_inputs.dim() == 4:  # (B, C, H, W)
            adjacent_scaled = F.interpolate(
                adjacent_inputs, size=disp.shape[-2:], mode="bilinear", align_corners=False
            )
            adjacent_scaled = adjacent_scaled.unsqueeze(1)  # Add a time-step dimension for consistency
        else:
            raise ValueError(f"adjacent_inputs must be 4D or 5D, but got {adjacent_inputs.dim()}D tensor.")

        # Compute photometric loss
        photometric_loss = compute_photometric_loss(target_scaled, adjacent_scaled, 
                                                    depth, pose_outputs, intrinsics)

        # Compute smoothness loss
        smoothness_loss = compute_smoothness_loss(disp, target_scaled)

        # Combine losses for the current scale
        scale_loss = photometric_weight * photometric_loss.mean() + smoothness_weight * smoothness_loss
        total_loss += scale_loss / len(scales)

    return total_loss


# Calculates photometric reconstruction loss using SSIM and L1 loss
def compute_photometric_loss(target, adjacent_frames, depth, pose_outputs, 
                             intrinsics):
    """
    Computes the photometric loss between the target frame and reconstructed frames.
    """
    batch_size, num_frames, _, height, width = adjacent_frames.shape
    photometric_loss = 0

    for i in range(num_frames):
        adjacent_frame = adjacent_frames[:, i]  # (B, C, H, W)
        pose = pose_outputs[:, i]              # (B, 6)

        # Reconstruct target frame from adjacent frame
        reconstructed_frame = reconstruct_image(target, depth, pose, intrinsics)

        # Compute L1 + SSIM loss
        l1_loss = torch.abs(target - reconstructed_frame)
        ssim_loss = ssim(target, reconstructed_frame)
        frame_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        # Add auto-masking
        automask = automask_loss(target, reconstructed_frame)
        frame_loss *= automask

        photometric_loss += frame_loss.mean()

    return photometric_loss / num_frames


# Encourages smooth disparity maps while preserving edges
def compute_smoothness_loss(disp, image):
    """
    Computes the edge-aware smoothness loss for the disparity map.
    """
    disp_grad_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    disp_grad_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    image_grad_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
    image_grad_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

    disp_grad_x *= torch.exp(-image_grad_x)
    disp_grad_y *= torch.exp(-image_grad_y)

    return disp_grad_x.mean() + disp_grad_y.mean()


# Reconstructs target image using adjacent frames, depth, and pose
def reconstruct_image(target, depth, pose, intrinsics):
    """
    Reconstruct an image using predicted depth and pose.

    Parameters:
        target (torch.Tensor): Target image (B, C, H, W).
        depth (torch.Tensor): Predicted depth map (B, 1, H, W).
        pose (torch.Tensor): Relative pose (B, 6) or (B, 2, 3).
        intrinsics (torch.Tensor): Camera intrinsics (B, 3, 3).

    Returns:
        torch.Tensor: Reconstructed image.
    """
    B, _, H, W = target.shape

    # Create a pixel grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=target.device, dtype=torch.float32),
        torch.arange(0, W, device=target.device, dtype=torch.float32),
        indexing="ij"
    )

    # Stack grid into homogeneous coordinates
    pixel_coords = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)  # (3, H, W)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 3, H, W)

    # Invert intrinsics
    inv_intrinsics = torch.inverse(intrinsics.squeeze(1))  # (B, 3, 3)

    # Transform pixel coordinates to camera coordinates
    cam_coords = depth * torch.einsum('bij,bjhw->bihw', inv_intrinsics, pixel_coords)

    # Debugging: Check pose shape
    # print(f"Pose shape before reshaping: {pose.shape}, total elements: {pose.numel()}")

    # Handle pose format
    if pose.dim() == 2 and pose.shape[1] == 6:  # Flattened pose format (B, 6)
        pose = pose.view(B, 2, 3)  # Reshape to (B, 2, 3)

    # Add the missing row [0, 0, 1] to make it (B, 3, 3)
    if pose.shape[1:] == (2, 3):  # Handle incomplete pose format
        bottom_row = torch.tensor([0, 0, 1], device=pose.device, dtype=pose.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        bottom_row = bottom_row.repeat(B, 1, 1)  # (B, 1, 3)
        pose = torch.cat([pose, bottom_row], dim=1)  # Now pose is (B, 3, 3)

    # Debugging: Check expanded pose shape
    # print(f"Pose shape after final expansion: {pose.shape}")

    # Extract rotation (R) and translation (t)
    R = pose[:, :, :3]  # Rotation: (B, 3, 3)
    t = torch.zeros((B, 3, 1), device=pose.device, dtype=pose.dtype)  # No explicit translation provided

    # Debugging: Check shapes of R, t, and cam_coords
    # print(f"R shape: {R.shape}, t shape: {t.shape}, cam_coords shape: {cam_coords.shape}")

    # Apply pose transformation
    cam_coords = cam_coords.view(B, 3, -1)  # Reshape to (B, 3, H*W)
    transformed_coords = R.bmm(cam_coords) + t  # (B, 3, H*W)
    transformed_coords = transformed_coords.view(B, 3, H, W)

    # Project back to image space
    proj_coords = intrinsics.squeeze(1).bmm(transformed_coords.view(B, 3, -1))
    proj_coords = proj_coords.view(B, 3, H, W)
    proj_coords = proj_coords[:, :2] / proj_coords[:, 2:3]  # Normalize by depth

    # Create the sampling grid
    grid = torch.stack([
        2.0 * proj_coords[:, 0] / (W - 1) - 1.0,  # Normalize x-coordinates
        2.0 * proj_coords[:, 1] / (H - 1) - 1.0   # Normalize y-coordinates
    ], dim=-1)  # (B, H, W, 2)

    # Clamp grid values to prevent out-of-bounds sampling
    grid = torch.clamp(grid, -1, 1)

    # Debugging: Check grid shape after permutation
    # print(f"Grid shape: {grid.shape}")  # Should be (B, H, W, 2)

    # Sample the reconstructed image
    reconstructed_image = F.grid_sample(target, grid, align_corners=False)

    return reconstructed_image


# SSIM function
def ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


# Filters out inconsistent or invalid pixels during photometric loss computation
def automask_loss(target, reconstructed):
    """
    Applies auto-masking to ignore inconsistent pixels.
    """
    photometric_diff = torch.abs(target - reconstructed)
    no_motion_mask = photometric_diff.mean(dim=1, keepdim=True) < photometric_diff.mean(dim=(1, 2, 3), keepdim=True)
    return no_motion_mask.float()

# Saving trained models
def save_model(encoder, depth_decoder, pose_encoder, pose_decoder, model_name, 
               epoch, save_folder_root):
    """
    Saves model checkpoints for monocular video training.

    Args:
        encoder (torch.nn.Module): Depth encoder network.
        depth_decoder (torch.nn.Module): Depth decoder network.
        pose_encoder (torch.nn.Module): Pose encoder network.
        pose_decoder (torch.nn.Module): Pose decoder network.
        model_name (str): Name of the model for saving.
        epoch (int): Current epoch number.
    """
    # Create directory for saving model checkpoints
    save_folder = os.path.join(save_folder_root, model_name)
    os.makedirs(save_folder, exist_ok=True)

    # Save encoder
    torch.save(encoder.state_dict(), os.path.join(save_folder, f"encoder_epoch_{epoch}.pth"))
    torch.save(depth_decoder.state_dict(), os.path.join(save_folder, f"depth_decoder_epoch_{epoch}.pth"))
    torch.save(pose_encoder.state_dict(), os.path.join(save_folder, f"pose_encoder_epoch_{epoch}.pth"))
    torch.save(pose_decoder.state_dict(), os.path.join(save_folder, f"pose_decoder_epoch_{epoch}.pth"))

    print(f"Model saved at epoch {epoch} in folder: {save_folder}")
    
# Validate trained models
def validate_model(encoder, depth_decoder, pose_encoder, pose_decoder, val_loader, device):
    """
    Validates the Monodepth2 model on the validation dataset for monocular videos.

    Args:
        encoder (torch.nn.Module): Depth encoder network.
        depth_decoder (torch.nn.Module): Depth decoder network.
        pose_encoder (torch.nn.Module): Pose encoder network.
        pose_decoder (torch.nn.Module): Pose decoder network.
        val_loader (DataLoader): Validation DataLoader.
        device (str): Device to run the validation (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average validation loss.
    """
    print("Starting validation...")
    encoder.eval()
    depth_decoder.eval()
    pose_encoder.eval()
    pose_decoder.eval()

    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Load validation data
            target_image = batch["image"].to(device)               # Target frame (B, 3, H, W)
            adjacent_images = batch["adjacent_images"].to(device)  # Adjacent frames (B, N, 3, H, W)
            intrinsics = batch["intrinsics"].to(device)            # Camera intrinsics (B, 3, 3)

            # Depth prediction
            features = encoder(target_image)
            outputs = depth_decoder(features)
            
            # Prepare pose_inputs by concatenating target image and first adjacent frame
            pose_inputs = torch.cat([target_image, adjacent_images[:, 0]], dim=1)  # (B, 6, H, W)

            # Debug: Print shape of pose_inputs
            # print(f"pose_inputs shape (before passing to pose_encoder): {pose_inputs.shape}")

            # Pose prediction
            pose_features = pose_encoder(pose_inputs)
            pose_outputs = pose_decoder(pose_features)

            # Compute validation loss
            loss = compute_depth_loss(target_image, adjacent_images, outputs, pose_outputs, intrinsics)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation completed. Average Loss: {avg_loss:.4f}")

    return avg_loss


# Visualize depth predictions for debugging
def visualize_depth_predictions(target_image, outputs, save_dir, batch_idx=0):
    """
    Visualizes and saves depth predictions for debugging.

    Args:
        target_image (torch.Tensor): Target input image (B, C, H, W).
        outputs (dict): Predicted outputs from the depth decoder.
        save_dir (str): Directory to save the visualizations.
        batch_idx (int): Index of the current validation batch for naming.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert predicted disparity to a normalized depth map
    disparity = outputs[("disp", 0)].squeeze().cpu().detach().numpy()
    depth = 1.0 / (disparity + 1e-7)  # Convert disparity to depth
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    # Convert target image to numpy for visualization
    target_image_np = target_image[0].permute(1, 2, 0).cpu().numpy()

    # Save visualizations
    target_image_path = os.path.join(save_dir, f"batch_{batch_idx}_target_image.png")
    depth_image_path = os.path.join(save_dir, f"batch_{batch_idx}_predicted_depth.png")

    plt.figure(figsize=(12, 6))

    # Target image
    plt.subplot(1, 2, 1)
    plt.title("Target Image")
    plt.imshow(target_image_np)
    plt.axis("off")
    plt.savefig(target_image_path, bbox_inches="tight")
    print(f"Saved target image to {target_image_path}")

    # Predicted depth
    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth")
    plt.imshow(depth_normalized, cmap="plasma")
    plt.axis("off")
    plt.savefig(depth_image_path, bbox_inches="tight")
    print(f"Saved predicted depth to {depth_image_path}")

    plt.close()