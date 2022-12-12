# Super resolution on mobile devices

## Contributors

* Jinam Shah
* Saumil Shah
  
## Repository Layout

All the files are present in the parent folder. The models that we generated at each step (vanilla model, unpruned model with optimal parameters, pruned model and the model with knowledge distilled) are present in the `generated_models` folder. Log files for all our experiments are present in the `logs` folder.

Please refer to the `XGen_Run_Results.csv` and `complete_experiment_v7.out` log files for our final experiment results.

The dataset used for these experiments in the div2k dataset. The dataset is present on our google drive [(link)](https://drive.google.com/drive/u/0/folders/0AFsxkLu_BbZoUk9PVA) and access for it can be shared if requested.

>Note: There are some code files that are kept for our reference and are not used in the current implementation, for example, `pruning_old.py`

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

| Model | Stage Quality (PSNR) | Speed (RF8M21Y9MNR device) | Speed (R38M20BDTME device) | Size (KB) |
| ---- | ---- | ---- | ---- | ---- |
| Original Unpruned model | 31.059138 | 8.951 | 8.951 | 2540 |
| Optimal HPO model | 30.748214 | 8.798 | 8.441 | 2540 |
| LevelPruned model | 28.699187 | 6.992 | 6.855 | 972 |
| Knowledge Distilled Pruned model | 30.222486 | 6.729 | 6.597 | 972 |

## Manual validation

We test our data on the Set5 dataset. The location of the image(s) to test is specified in the test_data_dir variable in `config.py` file. For validating results manually, please change this to the desired folder and use the `validation_runner.sh` shell script to validate results from all the models. The resulting high-resolution images will be available in `results/test` folder and are in a different folder for each stage.

> Note: The final models creating during our experiments and the resulting images from them are already available for reference. They are in `generated_models` and `results/test` folders respectively. The final log files are `logs/complete_experiment_v7.out` and `XGen_Run_Results.csv` for the model training/pruning/distillation and onnx latency check respectively.