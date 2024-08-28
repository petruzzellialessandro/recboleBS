import os 
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm

from recbole.quick_start import load_data_and_model
import torch
from recbole.utils.case_study import full_sort_scores, full_sort_topk


datasets = ['movielens', 'dbbook', 'lastfm']

sets_ = [set(os.listdir(f'out_{dataset}')) for dataset in datasets]
models = sets_[0]
for set_ in sets_:
    models = models & set_


os.environ["CUDA_VISIBLE_DEVICES"]="4, 6"
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-d", "--dataset", help = "Select Dataset", default='dbbook')
parser.add_argument("-t", "--testOnly", help = "Only test set items", default="True")
parser.add_argument("-k", "--k", help = "k top recommendation", default=-1, type=int)
 
# Read arguments from command line
args = parser.parse_args()
 
dataset_name = args.dataset
testOnly = eval(args.testOnly)
k = int(args.k)

if testOnly:
    k = -1 

for model_file in tqdm(os.listdir(f'saved_{dataset_name}')):
    os.makedirs(f'predicted_{dataset_name}', exist_ok=True)
    model_name = model_file.split("-")[0]
    #if model_name in [x.split('_')[-1].split('.')[0] for x in os.listdir(f'predicted_{dataset_name}')]:
    #    continue
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(os.path.join(f'saved_{dataset_name}', model_file))
    #model.to('cpu')
    if not testOnly:
        try:
            topk_score, topk_iid_list = full_sort_topk(
            test_data.uid_list, model, test_data, k=k, device=config["device"]
            )
            topk_iid_list = test_data.dataset.id2token(test_data.iid_field, topk_iid_list.detach().cpu().numpy())
            topk_uid_list = test_data.dataset.id2token(test_data.uid_field, test_data.uid_list)
            predicted_scores = topk_score.detach().cpu().numpy()
            out_list = [{'user_id': u, 'item_id': i, 'score':s} for u, i, s in zip(topk_uid_list, topk_iid_list, predicted_scores)]
            out_list = pd.DataFrame(out_list)
            out_list['score'].fillna(0, inplace=True)
            out_list.to_csv(os.path.join(f'predicted_{dataset_name}', f'predicted_scores_{model_name}_all_items.tsv'), index=None, sep='\t')
        except Exception:
            with open(os.path.join(f'predicted_{dataset_name}', f'ERROR_FILE.txt'), "a") as f:
                f.write(f"ERROR MODEL {model_name}\n")

    else:
        try:
            topk_score, topk_iid_list = full_sort_topk(
            test_data.uid_list, model, test_data, k=train_data.dataset.item_num, device=config["device"]
            )
            users, items = test_data.dataset.inter_matrix().nonzero()
            matrix_inter = torch.zeros((max(users), max(items)+1))
            for x, y in zip(users, items):
                matrix_inter[x-1][y] = 1
            items_ids = topk_iid_list[matrix_inter==1]
            predicted_scores = topk_score[matrix_inter==1]
            topk_iid_list = test_data.dataset.id2token(test_data.iid_field, items_ids.detach().cpu().numpy())
            topk_uid_list = test_data.dataset.id2token(test_data.uid_field, users)
            predicted_scores = predicted_scores.detach().cpu().numpy()
            out_list = [{'user_id': u, 'item_id': i, 'score':s} for u, i, s in zip(topk_uid_list, topk_iid_list, predicted_scores)]
            out_list = pd.DataFrame(out_list)
            out_list['score'].fillna(0, inplace=True)
            out_list.to_csv(os.path.join(f'predicted_{dataset_name}', f'predicted_scores_{model_name}.tsv'), index=None, sep='\t')
        except Exception:
            with open(os.path.join(f'predicted_{dataset_name}', f'ERROR_FILE.txt'), "a") as f:
                f.write(f"ERROR MODEL {model_name}\n")