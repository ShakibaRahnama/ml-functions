import os
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from mono_dataset_custom import MonoDataset
from networks import ResnetEncoder, DepthDecoder, PoseDecoder
from utils_custom import compute_depth_loss, visualize_depth_predictions
import wandb

# Ensure reproducibility
seed_everything(42, workers=True)

# Configuration
class Config:
    data = 'all_shortened_skip_10'   # 'subset' or 'all'
    frame_crop = 'cropped'   # 'cropped' or 'uncropped'
    data_path = f"/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Output_frames_split/{data}/{frame_crop}"
    train_file = f"{data_path}/train.txt"
    test_file = f"{data_path}/test.txt"
    height = 192
    width = 640
    batch_size = 16   # Total batch size
    num_epochs = 20
    learning_rate = 5e-5
    num_workers = 4
    log_frequency = 10
    save_frequency = 10
    model_name = f"pt_endo_epochs_{num_epochs}_batch_{batch_size}_lr_{learning_rate}_epoch_15_lr_{float(learning_rate/5)}_epoch_20_{data}_aug"

config = Config()

# Lightning Module
class MonoDepth2_Lightning(LightningModule):
    def __init__(self, config, train_loader, test_loader):
        super().__init__()
        self.config = config
        self.train_loader = train_loader  # Save training DataLoader
        self.test_loader = test_loader  # Save test DataLoader for validation

        # Initialize models
        self.depth_encoder = ResnetEncoder(18, pretrained=False)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, scales=range(4))
        self.pose_encoder = ResnetEncoder(18, pretrained=False, num_input_images=2)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

        # Load pretrained weights
        pretrained_paths = {
            "depth_encoder": "/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Pretrained_models/endodepth/epoch8/encoder.pth",
            "depth_decoder": "/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Pretrained_models/endodepth/epoch8/depth.pth",
            "pose_encoder": "/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Pretrained_models/endodepth/epoch8/pose_encoder.pth",
            "pose_decoder": "/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Pretrained_models/endodepth/epoch8/pose.pth",
        }

        self.depth_encoder.load_state_dict(torch.load(pretrained_paths["depth_encoder"], weights_only=True), strict=False)
        self.depth_decoder.load_state_dict(torch.load(pretrained_paths["depth_decoder"], weights_only=True), strict=False)
        self.pose_encoder.load_state_dict(torch.load(pretrained_paths["pose_encoder"], weights_only=True), strict=False)
        self.pose_decoder.load_state_dict(torch.load(pretrained_paths["pose_decoder"], weights_only=True), strict=False)

    def forward(self, target_image, adjacent_images):
        # Depth prediction
        features = self.depth_encoder(target_image)
        outputs = self.depth_decoder(features)

        # Pose prediction
        pose_inputs = torch.cat([target_image, adjacent_images[:, 0, :, :, :]], dim=1)
        pose_features = self.pose_encoder(pose_inputs)
        pose_outputs = self.pose_decoder(pose_features)

        return outputs, pose_outputs

    def train_dataloader(self):
        """Return training DataLoader."""
        return self.train_loader
    
    def training_step(self, batch, batch_idx):
        target_image = batch["image"]
        adjacent_images = batch["adjacent_images"]
        intrinsics = batch["intrinsics"]

        # Forward pass
        outputs, pose_outputs = self(target_image, adjacent_images)

        # Compute loss
        loss = compute_depth_loss(target_image, adjacent_images, outputs, 
                                  pose_outputs, intrinsics)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)

        # # Save depth maps for the first batch only
        # if batch_idx == 0:
        #     batch_size = target_image.shape[0]
        #     random.seed(42)  # Ensure reproducibility
        #     random_indices = random.sample(range(batch_size), k=min(5, batch_size))  # Select up to 10 samples
        #     print(f"Saving depth maps for random indices in batch {batch_idx}: {random_indices}")

        #     save_dir = f"/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/depth_visualizations/{self.config.model_name}/train/epoch_{self.current_epoch + 1}"
        #     os.makedirs(save_dir, exist_ok=True)

        #     for idx in random_indices:
        #         visualize_depth_predictions(
        #             target_image=target_image[idx:idx + 1],  # Select specific sample
        #             outputs={key: val[idx:idx + 1] for key, val in outputs.items()},  # Slice outputs
        #             save_dir=save_dir,
        #             batch_idx=idx,  # Use sample index within batch
        #         )
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        target_image = batch["image"]
        adjacent_images = batch["adjacent_images"]
        intrinsics = batch["intrinsics"]

        # Forward pass
        outputs, pose_outputs = self(target_image, adjacent_images)

        # Compute loss
        loss = compute_depth_loss(target_image, adjacent_images, outputs, pose_outputs, intrinsics)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Visualize predictions for randomly selected frames
        if hasattr(self, "random_indices") and batch_idx in self.random_indices:
            save_dir = f"/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/depth_visualizations/{self.config.model_name}/val/epoch_{self.current_epoch + 1}"
            os.makedirs(save_dir, exist_ok=True)
            visualize_depth_predictions(
                target_image=target_image,
                outputs=outputs,
                save_dir=save_dir,
                batch_idx=batch_idx,
            )

    def on_validation_start(self):
        """Hook to select random indices for visualization."""
        random.seed(42)
        # Use test_loader saved during initialization
        val_dataset = self.test_loader.dataset
        self.random_indices = random.sample(range(len(val_dataset)), k=10)

    def val_dataloader(self):
        """Return validation DataLoader."""
        return self.test_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        # Define lambda function for learning rate
        def lr_lambda(epoch):
            if epoch < 15:
                return 1.0  # Use initial learning rate (0.0005)
            else:
                return 0.2000  # drop by 5x

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',   # update learning rate every epoch
            }
        }
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), 
    #                                  lr=self.config.learning_rate)
    #     return optimizer


# Dataset Preparation
train_dataset = MonoDataset(
    data_path=config.data_path,
    filenames=open(config.train_file).read().splitlines(),
    height=config.height,
    width=config.width,
    is_train=True,
    num_adjacent_frames=1,
)
test_dataset = MonoDataset(
    data_path=config.data_path,
    filenames=open(config.test_file).read().splitlines(),
    height=config.height,
    width=config.width,
    is_train=False,
    num_adjacent_frames=1,
)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,   # set shuffle=False for debugging 
    num_workers=config.num_workers, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, 
    num_workers=config.num_workers, pin_memory=True, drop_last=False
)

# Callbacks and Logger
checkpoint_callback = ModelCheckpoint(
    dirpath=f"/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Models/{config.model_name}",
    save_top_k=1,  # Save only best epoch based on validation performance
    every_n_epochs=config.num_epochs,  # Save at last epoch
    save_last=True,  # Ensure last checkpoint is saved
)
lr_monitor = LearningRateMonitor(logging_interval="step")

# Custom Callback to Save Individual Model Weights
class SaveModelWeightsCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        # Define save directory
        save_dir = f"/cluster/projects/brudnogroup/ani/Bariatric_endoscopy/Models/{config.model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights as .pth files
        torch.save(pl_module.depth_encoder.state_dict(), f"{save_dir}/depth_encoder.pth")
        torch.save(pl_module.depth_decoder.state_dict(), f"{save_dir}/depth_decoder.pth")
        torch.save(pl_module.pose_encoder.state_dict(), f"{save_dir}/pose_encoder.pth")
        torch.save(pl_module.pose_decoder.state_dict(), f"{save_dir}/pose_decoder.pth")

        print(f"Model weights saved in {save_dir}")

# Instantiate the callback
save_model_weights_callback = SaveModelWeightsCallback()

# Initialize W&B
os.environ["WANDB_MODE"] = "offline"  # Ensure W&B operates in offline mode
wandb_logger = WandbLogger(
    project="Bariatric_Endoscopy",
    name=config.model_name,
    log_model=True,  # Optional: log model checkpoints as artifacts
    config={
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "image_height": config.height,
        "image_width": config.width,
    },
)

# Trainer
trainer = Trainer(
    max_epochs=config.num_epochs,
    callbacks=[checkpoint_callback, lr_monitor, save_model_weights_callback],
    devices="auto",  # automatically uses all node GPUs
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=True),  # Allow unused parameters
    logger=wandb_logger,
    log_every_n_steps=config.log_frequency,
)

# Train
model = MonoDepth2_Lightning(config, train_loader, test_loader)
trainer.fit(model)
