# Simplified Scripts for Baseline Evaluation in Recbole

## Environment Setup

### Singularity Container

This repository includes a `.def` file for building a Singularity container to ensure the correct Python version and necessary libraries are included.

#### Building the Container

To build the container on Unix systems, use the following command:
```sh
sudo singularity build recbole_container.sif recbole_container.def
```

After building the container, move the `.def` file to the current directory.

### Setting Up the Python Environment

These scripts are designed to work with **Python 3.10.12**. Using this version is strongly recommended for optimal compatibility.

#### Steps to Set Up the Environment

1. **Create a virtual environment**:
   ```sh
   python -m venv recboleEnv
   ```

2. **Activate the virtual environment**:
   - On Unix or macOS:
     ```sh
     source recboleEnv/bin/activate
     ```
   - On Windows:
     ```sh
     .\recboleEnv\Scripts\activate
     ```

3. **Install all dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset Setup

Place your dataset in the `dataset` folder within a subfolder named after the dataset. Each dataset folder must contain the following files:
- `{dataset_name}.part1.inter`: Contains the training set instances.
- `{dataset_name}.part3.inter`: Contains the test set instances.
- `{dataset_name}.item`: Contains the list of items and the corresponding features.
- `{dataset_name}.kg`: Contains the triples of the knowledge graph related to the items.
- `{dataset_name}.link`: Contains the mapping between the IDs used in the `kg` file and the `item` file.

Each `.inter` file must have three columns (user, item, score) separated by a tab (`\t`). The column names are not important, but the order of the columns is mandatory.

The `.kg` and `.link` files are optional. Refer to the Recbole documentation [here](https://recbole.io/docs/user_guide/data/atomic_files.html) for more information on file formats and requirements.

## Hyperparameter Optimization

The `hyperTuning` folder contains all the necessary code for hyperparameter tuning of the models implemented in Recbole. In this folder, you will find three `.py` files:

- `utils.py`: Contains the models that need to be optimized. For each model, it specifies the hyperparameter space to search for the best configuration. You can add new models and reference them using the class name in Recbole.

- `run_optim.py`: The script that runs the optimization. The script requires the following arguments:
  - `-d` or `--dataset`: The name of the dataset (folder name).
  - `-t` or `--trials`: The number of trials per model (default=50).
  - `-e` or `--early_stop`: The number of epochs for early stopping (default=10).

  Other settings, such as epochs and batch size, can be adjusted via the `config.yaml` file in the `config` folder.

  This script generates two types of files in a folder named `out_{dataset_name}`:
  1. `best_param_{model_name}.json`: A dictionary containing the best configuration of that model.
  2. `.ERROR_FILE.txt`: A list of models that encountered exceptions during the optimization phase.

### Example Usage

To run the optimization script with a dataset named `movielens`, 30 trials per model, and 100 epochs per trial, use:

```sh
singularity run --nv recbole_container.sif python run_optim.py -d movielens -t 30 -e 100
```

or 

```sh
python run_optim.py -d movielens -t 30 -e 100
```

## Train Model

The `train_save_model.py` script trains and saves models. It requires the file `{dataset_name}_optim_model.pkl`, which contains the model configurations. For each model in this file, the script will train and save it in a folder named `{dataset_name}_models`. Each model will be stored in its respective subfolder as a `.pkl` file.

This script requires three parameters:
- `-d` or `--dataset`: The dataset to use.
- `-e` or `--epochs`: The number of epochs per model (default=50).
- `-r` or `--rating_threshold`: The rating threshold for a "like" (default=1.0).

### Example Usage

To train and save models for a dataset named `movielens`, with 100 epochs per model and a rating threshold of 1.0, use:

```sh
singularity run --nv recbole_container.sif python train_save_model.py -d movielens -e 100 -r 1.0
```

or

```sh
python train_save_model.py -d movielens -e 100 -r 1.0
```

## Model Inference

The `inference_bs.py` script performs inference using trained Recbole models. It generates recommendation scores for users in the test set and saves the results to a file. This script requires three parameters:

- `-d` or `--dataset`: The name of the dataset.
- `-t` or `--testOnly`: If `True`, only items in the test set are considered for recommendations (default: `False`).
- `-k` or `--k`: The number of top recommendations to generate per user (default: `-1` for all items).

### Example Command

To run inference for a dataset named `movielens`, considering only test set items and generating the top 10 recommendations per user:

```sh
singularity run --nv recbole_container.sif python inference_bs.py -d movielens -t True -k 10
```
or
```sh
python inference_bs.py -d movielens -t True -k 10
```

This script generates two files:
1. `predicted_score_{model_name}.tsv`: Contains the recommended items for each user in the test set.
2. `predicted_score_{model_name}_all_items.tsv`: Contains all recommended items for each user if `testOnly` is set to `False`.

This script ensures that recommendations are generated efficiently and saved appropriately for further analysis or evaluation.
