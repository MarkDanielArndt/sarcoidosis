import torch
import argparse
import pathlib
import wandb

parser = argparse.ArgumentParser(description="Options for the run.")

parser.add_argument("--cluster", default=False, action="store_true")
parser.add_argument("--num_epochs", required=False, default=50)
parser.add_argument("--num_splits", required=False, default=10)
parser.add_argument("--num_nodes", required=False, default=800)


#parser.add_argument("--check_steps", required=False, default=10)
#parser.add_argument("--load_model", required=False, action="store_true", default=False)
#parser.add_argument("--save_model", required=False, action="store_true", default=True)

args = parser.parse_args()
cluster = args.cluster
num_epochs = int(args.num_epochs)
num_splits = int(args.num_splits)
num_nodes = int(args.num_nodes)

# Data paths
if cluster:
    folder_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "sarcoidosis" / 'sarcoidosis_MIPs_(npy)'
    csv_file_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "sarcoidosis" / "sacoidosis.CSV"
    checkpoint_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t" / "sarcoidosis" / "checkpoints" 
    image_path = pathlib.Path("/omics") / "groups" / "OE0471" / "internal" / "m623t"  / "sarcoidosis" / "plots"
    
else:
    folder_path = 'sarcoidosis_MIPs_(npy)'
    csv_file_path = 'sacoidosis.csv'
    image_path = pathlib.Path.cwd() / 'data' / 'plots' 
    checkpoint_path = pathlib.Path.cwd() / 'data' 

# Training parameters
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
batchsize = 32
n_splits = num_splits
learning_rate = 0.0001

# Images
new_height=400 
new_width=200
crop_x_start, crop_x_end = 60, 300 
crop_y_start, crop_y_end = 20, 200 

device = "cuda" if torch.cuda.is_available() else "cpu"

#wandb.init(
#    # set the wandb project where this run will be logged
#    project="sacoidosis",
#
#    # track hyperparameters and run metadata
#    config={
#    "learning_rate": learning_rate,
#    "architecture": "VGG16",
#    "dataset": "learning_rate",
#    "epochs": str(num_epochs),
#    }
#)
