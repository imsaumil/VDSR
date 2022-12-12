# Super resolution on mobile devices

## Contributors

* Jinam Shah
* Saumil Shah
  
## Repository Layout

All the files are present in the parent folder. The models that we generated at each step (vanilla model, unpruned model with optimal parameters, pruned model and the model with knowledge distilled) are present in the `generated_models` folder. Log files for all our experiments are present in the `logs` folder.

Please refer to the `onnx_latency3.csv` and `complete_experiment_v5.out` log files for our final experiment results.

The dataset used for these experiments in the div2k dataset. The dataset is present on our google drive [(link)](https://drive.google.com/drive/u/0/folders/0AFsxkLu_BbZoUk9PVA) and access for it can be shared if requested.

## VDSR model

For the VDSR model, there are a few model optimization techniques that we tried. Namely, pruning, distillation and hyper-parameter optimization.

## Execution

> Note: The scripts can be run directly using bash or slurm job manager.

For evaluating the project, a few steps will need to be taken.

**Dataset generation**

First, using the `dataset_runner.sh` script, generate the low-resolution/high-resolution pairs for training the model. Changes will need to be made to the script to work following the dataset's location.

Once this is done, take a subset of 2000 training image pairs to perform hyper-parameter optimization (HPO) and a subset of 20000 images for training the optimal model.

**Hyper parameter optimization**

Once the data subset for HPO is ready, define the data paths in `config.py` and run the `hpo_runner.sh` script.

**Model pruning and knowledge distillation**

The optimal parameters can be identified using the NNI WebUI and once that is done, add them to the `params` variable in the `pruning.py` and `distillation.py` files.

Finally, use the `train_runner.sh` shell script to run both model pruning and distillation.

Following the aforementioned steps, the models from each stage will be added to the `generated_models` folder.

**XGen**

The steps to perform xgen compatibility test and the onnx latency check on android devices is provided by Dr. Shen [(link)](https://docs.google.com/document/d/1guld2E_q42j7scS2lIHnSwJW8pGnTwW1xgOK_QAdi88/edit).

We perform the xgen checks on the server provided by Dr. Shen and on the generated onnx models. 


## Results

| Model | Stage	Quality (PSNR)	| Speed (RF8M21Y9MNR device) | Speed (R38M20BDTME device) | Size (KB) |
| ---- | ---- | ---- | ---- | ---- |
| Original Unpruned model | 31.05913811 | 204.961 | 204.047 | 2700 |
| Optimal HPO model	| 29.87049103 | 204.928 | 204.578 | 2700 |
| FPGMPruned model | 28.4406353 | 89.955 | 89.646 | 996 |
| Knowledge Distilled Pruned model | 28.44063911 | 90.058 | 89.309 | 996 |