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
parser.add_argument("-d", "--dataset", help = "Select Dataset", default='lastfm')
parser.add_argument("-t", "--testOnly", help = "Only test set items", default="True")
parser.add_argument("-k", "--k", help = "k top recommendation", default=-1, type=int)
 
# Read arguments from command line
args = parser.parse_args()
 
dataset_name = args.dataset
testOnly = eval(args.testOnly)
k = int(args.k)

if testOnly:
    k = -1 
out_folder = f'predicted_{dataset_name}_temp'

for model_file in tqdm(os.listdir(f'saved_{dataset_name}_temp')):
    os.makedirs(out_folder, exist_ok=True)
    model_name = model_file.split("-")[0]
    if model_name in [x.split('_')[-1].split('.')[0] for x in os.listdir(f'predicted_{dataset_name}_temp')] + ['ERROR_FILE.txt']:
        continue
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(os.path.join(f'saved_{dataset_name}_temp', model_file))
    #model.to('cpu')
    if hasattr(test_data, '_dataset'):
        testdts = test_data._dataset
        traindts = train_data._dataset
    else:
        testdts = test_data.dataset
        traindts = train_data.dataset
        #try:
    if not testOnly:
        topk_score, topk_iid_list = full_sort_topk(
        test_data.uid_list, model, test_data, k=k, device=config["device"]
        )
        topk_iid_list = testdts.id2token(test_data.iid_field, topk_iid_list.detach().cpu().numpy())
        topk_uid_list = testdts.id2token(test_data.uid_field, test_data.uid_list)
        predicted_scores = topk_score.detach().cpu().numpy()
        out_list = [{'user_id': u, 'item_id': i, 'score':s} for u, i, s in zip(topk_uid_list, topk_iid_list, predicted_scores)]
        out_list = pd.DataFrame(out_list)
        out_list['score'].fillna(0, inplace=True)
        out_list.to_csv(os.path.join(out_folder, f'predicted_score_{model_name}_all_items.tsv'), index=None, sep='\t')
    else:
        try:
            users, items = testdts.inter_matrix().nonzero()
            topk_score, topk_iid_list = full_sort_topk(
            test_data.uid_list, model, test_data, k=traindts.item_num, device=config["device"]
            )
            out_list = []
            for i_u, u in enumerate(test_data.uid_list):
                items_user = items[users==u]
                for item_ in items_user:
                    item_out = topk_iid_list[i_u][topk_iid_list[i_u] == item_].cpu().detach().numpy().item()
                    score_out = topk_score[i_u][topk_iid_list[i_u] == item_].cpu().detach().numpy().item()
                    out_list.append({'user_id': u, 'item_id': item_out, 'score':score_out})
            out_list = pd.DataFrame(out_list)
            out_list['user_id'] = out_list['user_id'].apply(int)
            out_list['item_id'] = out_list['item_id'].apply(int)
            out_list['user_id'] = testdts.id2token(test_data.uid_field, out_list['user_id'])
            out_list['item_id'] = testdts.id2token(test_data.iid_field, out_list['item_id'])
            out_list['score'].fillna(0, inplace=True)
            out_list.sort_values(['user_id', 'score'], ascending=False, inplace=True)
            out_list.to_csv(os.path.join(out_folder, f'predicted_score_{model_name}.tsv'), index=None, sep='\t')
        except Exception:
            with open(os.path.join(out_folder, f'ERROR_FILE.txt'), "a") as f:
                f.write(f"ERROR MODEL {model_name}\n")