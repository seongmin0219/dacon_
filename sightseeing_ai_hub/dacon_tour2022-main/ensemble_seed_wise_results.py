import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


result_dir_list = ["model_seed_40/",
                   "model_seed_41/",
                   "model_seed_42/",
                   "model_seed_43/",
                   "model_seed_44/",
                  ]

################################################################
## input and output path setting
train_csv_path = r'train.csv'
test_csv_path = r'test.csv'
sample_submit_csv_path = r'sample_submission.csv'

ensemble_result_save_dir = f"seed_wise_ensemble"
test_submit_csv_path = os.path.join(ensemble_result_save_dir, "submit.csv")

#################################################################

if not os.path.exists(ensemble_result_save_dir):
    os.makedirs(ensemble_result_save_dir)
all_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)


le_3 = preprocessing.LabelEncoder()
le_3.fit(all_df['cat3'].values)
all_df['cat3'] = le_3.transform(all_df['cat3'].values)

pred_probs_cat1_seeds = np.zeros([len(test_df), 6])  ### cat1 class probability
pred_probs_cat2_seeds = np.zeros([len(test_df), 18])  ### cat2 class probability
pred_probs_cat3_seeds = np.zeros([len(test_df), 128])  ### cat3 class probability
pred_probs_ens_seeds = np.zeros([len(test_df), 128])  ### ensemble of cat1 cat2 cat3 probability

for result_dir in result_dir_list:
    pred_probs_cat1 = np.load(os.path.join(result_dir,'cat1_probs.npy'))
    pred_probs_cat2 = np.load(os.path.join(result_dir,'cat2_probs.npy'))
    pred_probs_cat3 = np.load(os.path.join(result_dir,'cat3_probs.npy'))
    pred_probs_ens = np.load(os.path.join(result_dir,'cat_ens_probs.npy'))


    pred_probs_cat1_seeds += pred_probs_cat1
    pred_probs_cat2_seeds += pred_probs_cat2
    pred_probs_cat3_seeds += pred_probs_cat3
    pred_probs_ens_seeds += pred_probs_ens

pred_probs_cat1_seeds = pred_probs_cat1_seeds/len(result_dir_list)
pred_probs_cat2_seeds = pred_probs_cat2_seeds/len(result_dir_list)
pred_probs_cat3_seeds = pred_probs_cat3_seeds/len(result_dir_list)
pred_probs_ens_seeds = pred_probs_ens_seeds/len(result_dir_list)


#### save ensemble results, cat1 cat2 cat3 probabilities will used to do further knowledge distillation
np.save(os.path.join(ensemble_result_save_dir, "cat1_probs.npy"), pred_probs_cat1_seeds)
np.save(os.path.join(ensemble_result_save_dir, "cat2_probs.npy"), pred_probs_cat2_seeds)
np.save(os.path.join(ensemble_result_save_dir, "cat3_probs.npy"), pred_probs_cat3_seeds)


#### seed wise test result ensemble

np.save(os.path.join(ensemble_result_save_dir, "cat_ens_probs.npy"), pred_probs_ens_seeds)
preds_ens_cat = np.argmax(pred_probs_ens_seeds, axis=1)
submit = pd.read_csv(sample_submit_csv_path)
submit['cat3'] = le_3.inverse_transform(preds_ens_cat)
submit.to_csv(test_submit_csv_path, index=False)





