import os
from datetime import datetime

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# total training epoches
EPOCH = 50
MILESTONES = [20, 40]

# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().isoformat()

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10