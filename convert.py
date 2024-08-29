import os
import pandas as pd
from tqdm import tqdm

base_path = 'predicted_movielens'

files_ = os.listdir(base_path)#.remove('ERROR_FILE.txt')
os.makedirs(base_path, exist_ok=True)

for f in tqdm(files_):
    df = pd.read_csv(os.path.join(base_path,f), sep='\t')
    df['score'].fillna(0, inplace=True)
    df.sort_values(['user_id', 'score'], ascending=False).to_csv(os.path.join(base_path, f.replace('scores', 'score')), index=False, sep='\t')
