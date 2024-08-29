import os 
import re
import json 
import warnings
warnings.filterwarnings("ignore")
import argparse

from recbole.quick_start import run

datasets = ['movielens', 'dbbook', 'lastfm']

sets_ = [set(os.listdir(f'out_{dataset}')) for dataset in datasets]
models = sets_[0]
for set_ in sets_:
    models = models & set_


 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-d", "--dataset", help = "Select Dataset", default='lastfm')
 
# Read arguments from command line
args = parser.parse_args()
 
if args.dataset:
    dataset = args.dataset

models = models.difference(set("ERROR_FILE.txt"))
os.makedirs(f'saved_{dataset}_temp', exist_ok=True)
for model_file in models:
    match = re.search(r"best_param_(.*)\.json", model_file)
    if match:
        model_name = match.group(1)
    #if model_name in set([x.split('-')[0] for x in os.listdir(f'saved_{dataset}')]):
    #    continue
    try:
        with open(os.path.join(f'out_{dataset}', model_file)) as json_file:
            data = json.load(json_file)
            for k in set(data.keys()).difference(['dataset', 'model', 'benchmark_filename']):
                del data[k]
            data['checkpoint_dir']=f'saved_{dataset}_temp'
            run(model = data['model'], dataset = data['dataset'], config_dict=data, config_file_list=[os.path.join('configs', 'config.yaml')])
    except Exception:
        with open(os.path.join(f'saved_{dataset}_temp', f'ERROR_FILE.txt'), "a") as f:
            f.write(f"ERROR MODEL {model_name}\n")
    