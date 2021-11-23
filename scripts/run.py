import os

# Prepare dataset
os.system("python ./prepare_dataset.py")

# Split train and valid
os.system("python ./split_train_valid_dataset.py")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --image_dir TB291/VDSR/train/inputs --lmdb_path train_lmdb/VDSR/TB291_LR_lmdb")
os.system("python ./create_lmdb_dataset.py --image_dir TB291/VDSR/train/target --lmdb_path train_lmdb/VDSR/TB291_LR_lmdb")

os.system("python ./create_lmdb_dataset.py --image_dir TB291/VDSR/valid/inputs --lmdb_path valid_lmdb/VDSR/TB291_LR_lmdb")
os.system("python ./create_lmdb_dataset.py --image_dir TB291/VDSR/valid/target --lmdb_path valid_lmdb/VDSR/TB291_HR_lmdb")
