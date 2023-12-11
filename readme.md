## DiffCMR

Tianqi Xiang*, Wenjun Yue*, Yiqun Lin, Jiewen Yang, Zhenkun Wang, Xiaomeng Li,  ["DiffCMR: Fast Cardiac MRI Reconstruction with Diffusion Probabilistic Models"](https://arxiv.org/abs/2312.04853) MICCAI2023 CMRxRecon Workshop

### Data preparation

Download the CMRxRecon dataset from the [official website](https://cmrxrecon.github.io/Challenge.html).

#### Prepare Task 1 data

Open `data_task1.ipynb` and modify 2 variables:

```python
original_dataset_path = "/<your path>/MICCAIChallenge2023/ChallengeData"
save_prepared_dataset_path = "/<your path>/MICCAIChallenge2023/Task1"
```

Run `data_task1.ipynb` to get preprocessed data and 6 pair files for Task 1.

#### Prepare Task 2 data

Open `data_task2.ipynb` and modify 3 variables:

```python
dataroot = "/<your path>/MICCAIChallenge2023/ChallengeData"
saveroot = "/<your path>/MICCAIChallenge2023/Task2"
pair_dir = "/<your path>/MICCAIChallenge2023/Task2_pair"
```

Run `data_task2.ipynb` to get preprocessed data and pair files for Task 2.

Note that pair files are generated to record the absolute paths of preprocessed data, as long as the path for preprocessed data remains unchanged, you can relocate the pair files at your convience.

### Train

#### Task 1

Open `train_task1_diff.py`, modify the variable `trainpairfile` to the path of pair file for Task1 AccFactor 04/08/10.

```shell
python train_task1_diff.py
```

#### Task 2

Open `train_task2_diff.py`, modify the variable `trainpairfile` to the path of pair file for Task2 AccFactor 04/08/10.

```shell
python train_task2_diff.py
```

After training, model weights are stored in `logdir` set in the training files. Note that you will be running 6 times to conduct the entire experiment for both tasks and all coil numbers.

### Inference

Open `inferece.py`, modify 3 variables:

```python
result_dir = "<path to store results>"
model_path = "<path of model weights>"
val_pair_file = "<validation pair file>"
```

Execute `inferece.py`. Results are listed in the `result_dir`.
