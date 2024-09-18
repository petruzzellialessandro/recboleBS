# Simplified Scripts for Baseline Evaluation in Recbole

## Environment Setup

### Singularity Container

This repository includes a `.def` file to build a Singularity container, ensuring the correct Python version and necessary libraries are installed.

#### Building the Container

To build the container on Unix systems, run the following command:
```sh
sudo singularity build recbole_container.sif recbole_container.def
```

After building the container, move the `.def` file to the current directory.

### Python Environment Setup

The scripts are designed for **Python 3.10.12**. Using this version is strongly recommended for compatibility.

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

Place your dataset in the `dataset` folder within a subfolder named after the dataset. Each dataset folder must include the following files:
- `{dataset_name}.part1.inter`: Contains the training set instances.
- `{dataset_name}.part3.inter`: Contains the test set instances.
- `{dataset_name}.item`: Contains the list of items and their corresponding features.
- `{dataset_name}.kg`: Contains the triples of the knowledge graph related to the items.
- `{dataset_name}.link`: Maps the IDs used in the `kg` file to those in the `item` file.

Each `.inter` file must have three columns (user, item, score) separated by a tab (`\t`). While the column names can vary, the order must be maintained.

The `.kg` and `.link` files are optional. Refer to the Recbole documentation [here](https://recbole.io/docs/user_guide/data/atomic_files.html) for more details on file formats and requirements.

## Hyperparameter Optimization

The `hyperTuning` folder contains the necessary code for hyperparameter tuning of models implemented in Recbole. Inside this folder, you'll find three `.py` files:

- `utils.py`: Defines the models to be optimized, including the hyperparameter space for each model. New models can be added and referenced by their class names in Recbole.
- `run_optim.py`: Runs the optimization process. It accepts the following arguments:
  - `-d` or `--dataset`: The dataset name (folder name).
  - `-t` or `--trials`: The number of trials per model (default = 50).
  - `-e` or `--early_stop`: The number of epochs for early stopping (default = 10).

  Other settings, such as epochs and batch size, can be adjusted via the `config.yaml` file in the `config` folder.

  This script generates two types of files in a folder named `out_{dataset_name}`:
  1. `best_param_{model_name}.json`: Contains the best configuration for each model.
  2. `ERROR_FILE.txt`: Lists models that encountered exceptions during the optimization phase.

### Example Usage

To run the optimization script with a dataset named `movielens`, 30 trials per model, and 100 epochs per trial, use:

```sh
singularity run --nv recbole_container.sif python run_optim.py -d movielens -t 30 -e 100
```

or 

```sh
python run_optim.py -d movielens -t 30 -e 100
```

## Model Training

The `train_save_model.py` script trains and saves models. It requires the `out_{dataset_name}` folder, which contains the model configurations. For each model file in this folder, the script will train and save it in a folder named `saved_{dataset_name}`.

This script requires the following parameters:
- `-d` or `--dataset`: The dataset to use.

Other settings, such as epochs and batch size, can be adjusted via the `config.yaml` file in the `config` folder.

> **Note**: The script will automatically ignore the `ERROR_FILE.txt`. However, it may generate its own `ERROR_FILE.txt` in the `saved_{dataset_name}` folder, following the same structure.

### Example Usage

To train and save models for a dataset named `movielens`, use:

```sh
singularity run --nv recbole_container.sif python train_save_model.py -d movielens
```

or

```sh
python train_save_model.py -d movielens
```

## Model Inference

The `inference_bs.py` script performs inference using trained Recbole models, generating recommendation scores for users in the test set. It saves the results to a file and requires the following parameters:

- `-d` or `--dataset`: The dataset name.
- `-t` or `--testOnly`: If `True`, only items in the test set are considered for recommendations (default: `False`).
- `-k` or `--k`: The number of top recommendations to generate per user (default: `-1` for all items).

### Example Usage

To run inference for a dataset named `movielens`, considering only test set items and generating the top 10 recommendations per user:

```sh
singularity run --nv recbole_container.sif python inference_bs.py -d movielens -t True -k 10
```
or
```sh
python inference_bs.py -d movielens -t True -k 10
```

This script generates two files in a folder named `predicted_{dataset_name}`:
1. `predicted_score_{model_name}.tsv`: Contains the recommended items for each user in the test set.
2. `predicted_score_{model_name}_all_items.tsv`: Contains all recommended items for each user if `testOnly` is set to `False`.

This script ensures that recommendations are efficiently generated and saved for further analysis or evaluation.