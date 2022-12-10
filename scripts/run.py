import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir /ocean/projects/cis220070p/jshah2/div2k/original --output_dir /ocean/projects/cis220070p/jshah2/div2k/train --image_size 42 --step 42 --scale 2 --num_workers 40")
os.system("python3 ./prepare_dataset.py --images_dir /ocean/projects/cis220070p/jshah2/div2k/original --output_dir /ocean/projects/cis220070p/jshah2/div2k/train --image_size 42 --step 42 --scale 3 --num_workers 40")
os.system("python3 ./prepare_dataset.py --images_dir /ocean/projects/cis220070p/jshah2/div2k/original --output_dir /ocean/projects/cis220070p/jshah2/div2k/train --image_size 44 --step 44 --scale 4 --num_workers 40")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --train_images_dir /ocean/projects/cis220070p/jshah2/div2k/train --valid_images_dir /ocean/projects/cis220070p/jshah2/div2k/valid --valid_samples_ratio 0.1")
