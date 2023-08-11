from pytorch_lightning import Callback
from torchvision.utils import make_grid
import torch

class ImageLogger(Callback):
    def __init__(self, log_interval=1):
        super().__init__()
        self.log_interval = log_interval

    def on_epoch_end(self, trainer, pl_module):
        # Check if we should log this epoch
        if (trainer.current_epoch + 1) % self.log_interval == 0:
            log = pl_module.log_images(trainer.datamodule.val_dataloader())
            for key, images in log.items():
                # Convert the images to a grid
                grid = make_grid(images)
                # Log the images to TensorBoard
                trainer.logger.experiment.add_image(key, grid, global_step=trainer.global_step)