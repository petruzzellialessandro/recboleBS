import hyper.utils as utils
import os
from recbole.quick_start import objective_function

from recbole.trainer import HyperTuning
import warnings
warnings.filterwarnings("ignore")

import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-d", "--dataset", help = "Select Dataset")
parser.add_argument("-t", "--trails", help = "Number of trails", type=int, default=50)
parser.add_argument("-e", "--early_stop", help = "Number of early stop epochs", type=int, default=10)
 
# Read arguments from command line
args = parser.parse_args()
 
if args.dataset:
    dataset = args.dataset

max_iter = int(args.trails)
early_stop = int(args.early_stop)

fixed_config_file_list = [os.path.join('configs','config.yaml')]
os.makedirs(f'out_{dataset}', exist_ok=True)

for model_name, params_dict in utils.config_dict.items():
    try:
        params_dict['choice'] = params_dict.get('choice', {})
        params_dict['choice']['model'] = [model_name]
        params_dict['choice']['dataset'] = [dataset]
        params_dict['choice']['benchmark_filename'] = ['["part1", "part3", "part3"]']
        out_path = os.path.join(f'out_{dataset}', f'best_param_{model_name}.json')

        hp = HyperTuning(
                objective_function,
                algo="bayes",
                early_stop=early_stop,
                max_evals=max_iter,
                params_dict=params_dict,
                fixed_config_file_list=fixed_config_file_list
            )
        hp.run()

        import json

        json_data = json.dumps(hp.best_params, indent=4) 
        with open(out_path, "w") as f:
            f.write(json_data)
    except Exception:
        with open(os.path.join(f'out_{dataset}', f'ERROR_FILE.txt'), "a") as f:
            f.write(f"ERROR MODEL {model_name}\n")