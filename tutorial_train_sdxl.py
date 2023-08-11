from share import *
from scripts.streamlit_helpers import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import SDXLDataset
from cldmXL.logger import ImageLogger
from cldmXL.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# set checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='{epoch}-checkpoint',
    every_n_epochs=1,
    verbose=True,
)

# Configs
resume_path = './models/control_sdxl_ini.ckpt'
batch_size = 1
accumulate_grad_batches = 2
logger_freq = 500
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_xl.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
sampler, num_rows, num_cols = init_sampling(stage2strength=None)
model.sampler = sampler
model.model.half()

# Misc
dataset = SDXLDataset()
dataloader = DataLoader(dataset, num_workers=6, batch_size=batch_size, shuffle=True)
logger = ImageLogger(log_interval=2)
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1, precision=16, callbacks=[logger, checkpoint_callback],accumulate_grad_batches = accumulate_grad_batches, limit_train_batches=1.0, log_every_n_steps=1)


# Train!
trainer.fit(model, dataloader)
