import random
import pandas as pd
import numpy as np
import os
import copy
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings(action='ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

### train using this script with default setting needs 24GB gpu memory,
### if your device lack of gpu memory, you can reduce batch size or max word length
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



## utility functions
##############################################################################################################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#### matching cat1 label to corresponding cat3 label
#### matching cat2 label to corresponding cat3 label
#### using this matching pairs we can ensemble cat1 cat2 cat3 probability
def get_label_trees(df):
    cat1_tree_dict = {}
    cat2_tree_dict = {}
    for label_i in range(len(df["cat1"].unique())):
        cat1_tree_dict[label_i] = df[df["cat1"] == label_i]["cat3"].unique()

    for label_i in range(len(df["cat2"].unique())):
        cat2_tree_dict[label_i] = df[df["cat2"] == label_i]["cat3"].unique()

    return cat1_tree_dict, cat2_tree_dict

#### ensemble cat1, cat2, cat3, probability by paired label to get ensembled probability
def ensemble_cat123(output_1_mean, output_2_mean, output_3_mean, cat1_tree_dict, cat2_tree_dict, cat1_coef=1, cat2_coef=1, cat3_coef=1):
    output_1_mean = F.softmax(output_1_mean, dim=1) * cat1_coef
    output_2_mean = F.softmax(output_2_mean, dim=1) * cat2_coef
    output_3_mean = F.softmax(output_3_mean, dim=1) * cat3_coef
    for k, v in cat1_tree_dict.items():
        output_3_mean[:, v] += output_1_mean[:, [k]]

    for k, v in cat2_tree_dict.items():
        output_3_mean[:, v] += output_2_mean[:, [k]]
    return output_3_mean

def score_function(real, pred):
    return f1_score(real, pred, average="weighted")
############################################################################################################


### simple pytorch datasets class
### we only use text information
### cat1 cat2 labels are also used
class CustomDataset(Dataset):
    def __init__(self, df, infer=False):
        self.df = df
        self.txt_list = df['overview'].values
        if not infer:
            self.label_1_list = df['cat1'].values
            self.label_2_list = df['cat2'].values
            self.label_3_list = df['cat3'].values
        self.infer = infer

    def __getitem__(self, index):

        # Text
        text = self.txt_list[index]
        # Label
        if self.infer:
            return text
        else:
            label_1 = self.label_1_list[index]
            label_2 = self.label_2_list[index]
            label_3 = self.label_3_list[index]

            return text, label_1, label_2, label_3

    def __len__(self):
        return len(self.txt_list)


### dataset class for distillation
### get text information of test data and corresponding predicted probability,
### which yield using seed wise ensemble (train_and_infer.py, ensemble_seed_wise_results.py)

class CustomDataset_Distill(Dataset):
    def __init__(self, df, cat1_probs, cat2_probs, cat3_probs):
        self.df = df
        self.txt_list = df['overview'].values
        self.cat1_probs = cat1_probs
        self.cat2_probs = cat2_probs
        self.cat3_probs = cat3_probs

    def __getitem__(self, index):
        # Text
        text = self.txt_list[index]
        # Label
        cat1_prob = self.cat1_probs[index, :]
        cat2_prob = self.cat2_probs[index, :]
        cat3_prob = self.cat3_probs[index, :]
        return text, cat1_prob, cat2_prob, cat3_prob

    def __len__(self):
        return len(self.txt_list)


### model definition
### for each of cat1, cat2, cat3 prediction, we added two transformer encoder layer on klue/roberta-large network
class CustomModel(nn.Module):
    def __init__(self, txt_model, tokenizer, cat1_classes=6, cat2_classes=18, cat3_classes=128, max_length=128):
        super(CustomModel, self).__init__()

        ### txt_model used hugging face pretrained klue/roberta-large model
        self.txt_model = txt_model
        self.tokenizer = tokenizer
        self.hidden_size = self.txt_model.config.hidden_size
        self.max_length = max_length

        ### output cat1 probability
        self.classifier_1 = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True),
                                  num_layers=2),
            nn.Linear(self.hidden_size, cat1_classes),
        )

        ### output cat2 probability
        self.classifier_2 = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True),
                                  num_layers=2),
            nn.Linear(self.hidden_size, cat2_classes),
        )

        ### output cat3 probability
        self.classifier_3 = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True),
                                  num_layers=2),
            nn.Linear(self.hidden_size, cat3_classes),
        )

    def forward(self, txt, aug=True):
        txt = list(txt)

        ### text augmentation for train
        ### 50% to augment text
        ### randomly remove continous 25%~50% character
        txt_aug = []
        if aug:
            for txt_item in txt:
                if random.random() > 0.5:
                    remove_char_num = int(random.uniform(0.25, 0.5) * len(txt_item))
                    random_offset = random.randint(0, len(txt_item) - remove_char_num)
                    random_end = len(txt_item) - random_offset - remove_char_num
                    txt_item_aug = txt_item[:random_offset] + ' ' + txt_item[-random_end:]
                    txt_aug.append(txt_item_aug)
                else:
                    txt_aug.append(txt_item)
        else:
            txt_aug = txt

        txt_splited_all_encoded = self.tokenizer(txt_aug, max_length=self.max_length, add_special_tokens=True,
                                                 padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        txt_feature = self.txt_model(**txt_splited_all_encoded, output_hidden_states=True)

        ### get first [cls] token of the last 5th layer, using this feature to predict cat1 class
        txt_feature_1 = txt_feature.hidden_states[-5][:, 0, :]
        ### get first [cls] token of the last 3rd layer, using this feature to predict cat2 class
        txt_feature_2 = txt_feature.hidden_states[-3][:, 0, :]
        ### get first [cls] token of the last layer, using this feature to predict cat3 class
        txt_feature_3 = txt_feature.hidden_states[-1][:, 0, :]

        output_1 = self.classifier_1(txt_feature_1)
        output_2 = self.classifier_2(txt_feature_2)
        output_3 = self.classifier_3(txt_feature_3)

        return output_1, output_2, output_3


#### train pipeline
def train(model, optimizer, train_loader, distill_loader, val_loader, scheduler, cat1_tree_dict, cat2_tree_dict,
          model_save_dir,
          max_epoch=100, fold_i=0):
    model.to(DEVICE)
    distill_iter = iter(distill_loader)
    best_score = 0
    best_epoch = 0
    best_model = None

    ### scaler for mixed precision training, it can reduce nearly half of gpu memory
    ### gradient accumulation is also applied to reduce gpu memory
    scaler = torch.cuda.amp.GradScaler()
    iters_to_accumulate = 4
    for epoch in range(1, max_epoch + 1):
        model.train()
        train_loss = []
        for batch_i, (txt, label_1, label_2, label_3) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                label_1, label_2, label_3 = label_1.type(torch.LongTensor), label_2.type(
                    torch.LongTensor), label_3.type(torch.LongTensor)

                #### get distillation test sample batch
                try:
                    distill_txt, distill_cat1, distill_cat2, distill_cat3 = next(distill_iter)
                except StopIteration:
                    distill_iter = iter(distill_loader)
                    distill_txt, distill_cat1, distill_cat2, distill_cat3 = next(distill_iter)

                total_txt = list(txt)
                total_txt.extend(list(distill_txt))  #### concatenate train data and distillation test data

                output_1, output_2, output_3 = model(total_txt, aug=True)

                ### loss for cat1 cat2 cat3 train data
                loss_1 = F.cross_entropy(output_1[:-len(distill_txt), :], label_1.to(DEVICE))
                loss_2 = F.cross_entropy(output_2[:-len(distill_txt), :], label_2.to(DEVICE))
                loss_3 = F.cross_entropy(output_3[:-len(distill_txt), :], label_3.to(DEVICE))

                ### loss for distillation data, soft label cross entropy is applied,
                ### due to the label is not ground truth label, we applied 0.3 weight to reduce its importance to total loss
                distill_loss_1 = F.cross_entropy(output_1[-len(distill_txt):, :], F.softmax(distill_cat1.to(DEVICE), dim=1)) * 0.3
                distill_loss_2 = F.cross_entropy(output_2[-len(distill_txt):, :], F.softmax(distill_cat2.to(DEVICE), dim=1)) * 0.3
                distill_loss_3 = F.cross_entropy(output_3[-len(distill_txt):, :], F.softmax(distill_cat3.to(DEVICE), dim=1)) * 0.3

                ### total loss that combines distillation loss and loss for train data
                ### weighting loss for cat1->0.05 cat2->0.15 cat3->0.8
                total_loss = (loss_1 + distill_loss_1) * 0.05 + (loss_2 + distill_loss_2) * 0.15 + (loss_3 + distill_loss_3) * 0.8
                train_loss.append(total_loss.item())
                total_loss = total_loss / iters_to_accumulate

            scaler.scale(total_loss).backward()
            if (batch_i + 1) % iters_to_accumulate == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        tr_loss = np.mean(train_loss)
        if scheduler is not None:
            scheduler.step()

        ### get validation score
        val_loss, val_score = validation(model, val_loader, cat1_tree_dict, cat2_tree_dict)

        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')

        ### save model with best validation score for each fold
        if best_score < val_score:
            best_score = val_score
            best_model = copy.deepcopy(model)
            best_epoch = epoch
    obj = {
        'epoch': best_epoch,
        'model': best_model.state_dict(),
    }
    checkpoint_filename = f"fold_{fold_i + 1}_checkpoint_{best_epoch}_" \
                          f"vs_{best_score:.5f}.pth"
    torch.save(obj, os.path.join(model_save_dir, checkpoint_filename))

    return best_model


def validation(model, val_loader, cat1_tree_dict, cat2_tree_dict):
    model.eval()

    model_preds = []
    true_labels = []
    val_loss = []

    with torch.no_grad():
        for batch_i, (txt, label_1, label_2, label_3) in enumerate(val_loader):
            label_1, label_2, label_3 = label_1.type(torch.LongTensor), label_2.type(torch.LongTensor), label_3.type(
                torch.LongTensor)
            label_1, label_2, label_3 = label_1.to(DEVICE), label_2.to(DEVICE), label_3.to(DEVICE)

            output_1, output_2, output_3 = model(txt, aug=False)

            loss_1 = F.cross_entropy(output_1, label_1)
            loss_2 = F.cross_entropy(output_2, label_2)
            loss_3 = F.cross_entropy(output_3, label_3)

            total_loss = loss_1 * 0.05 + loss_2 * 0.15 + loss_3 * 0.8
            val_loss.append(total_loss.item())

            ### final prediction of the model is ensemble of cat1, cat2, cat3 probability
            ### cat1 probability is added to corresponding low-level cat3
            ### cat2 probability is added to corresponding low-level cat3
            output_ens = ensemble_cat123(output_1, output_2, output_3, cat1_tree_dict, cat2_tree_dict, cat1_coef=1,
                                         cat2_coef=1, cat3_coef=1)

            model_preds += output_ens.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label_3.detach().cpu().numpy().tolist()

    test_weighted_f1 = score_function(true_labels, model_preds)
    return np.mean(val_loss), test_weighted_f1

### inference for test data
def inference(model, test_loader, cat1_tree_dict, cat2_tree_dict):
    print(f"Test Step")
    model.to(DEVICE)
    model.eval()

    model_preds = []
    #### save predicted cat1 cat2 cat3 and ensemble prediction probability

    pred_probs_1 = np.zeros([len(test_loader.dataset), 6])
    pred_probs_2 = np.zeros([len(test_loader.dataset), 18])
    pred_probs_3 = np.zeros([len(test_loader.dataset), 128])
    pred_probs_ens = np.zeros([len(test_loader.dataset), 128])

    bs = test_loader.batch_size
    with torch.no_grad():
        for batch_i, txt in enumerate(test_loader):
            output_1, output_2, output_3 = model(txt, aug=False)
            ### final prediction of the model is ensemble of cat1, cat2, cat3 probability
            ### cat1 probability is added to corresponding low-level cat3
            ### cat2 probability is added to corresponding low-level cat3
            output_ens = ensemble_cat123(output_1, output_2, output_3, cat1_tree_dict, cat2_tree_dict, cat1_coef=1,
                                         cat2_coef=1, cat3_coef=1)

            pred_probs_1[batch_i * bs: batch_i * bs + bs, :] = output_1.detach().cpu().numpy()
            pred_probs_2[batch_i * bs: batch_i * bs + bs, :] = output_2.detach().cpu().numpy()
            pred_probs_3[batch_i * bs: batch_i * bs + bs, :] = output_3.detach().cpu().numpy()
            pred_probs_ens[batch_i * bs: batch_i * bs + bs, :] = output_ens.detach().cpu().numpy()

            model_preds += output_ens.argmax(1).detach().cpu().numpy().tolist()

    return model_preds, pred_probs_ens, pred_probs_1, pred_probs_2, pred_probs_3


def main():
    batch_size = 8
    num_workers = 4
    max_length = 256
    max_epoch = 31
    learning_rate = 0.00003
    seed=44  ### seed is fixed to 44
    ###################################################################################################################
    ## input output path setting and seed setting
    ###################################################################################################################

    train_csv_path = r'train.csv'   ### change to your path
    test_csv_path = r'test.csv'     ### change to your path
    sample_submit_csv_path = r'sample_submission.csv'   ### change to your path

    ### predicted probability saved dir, and distill model save dir
    predicted_probs_dir = f"seed_wise_ensemble"
    model_save_dir = f"seed_wise_ensemble"

    ###################################################################################################################
    ###################################################################################################################
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    ### final submission file path
    test_submit_csv_path = os.path.join(model_save_dir, "distill_submit_5fold.csv")
    seed_everything(seed)

    ### read predicted probability on test data
    distill_cat1 = np.load(os.path.join(model_save_dir, "cat1_probs.npy"))
    distill_cat2 = np.load(os.path.join(model_save_dir, "cat2_probs.npy"))
    distill_cat3 = np.load(os.path.join(model_save_dir, "cat3_probs.npy"))

    #### pretrained language model used in this competition
    #### https://huggingface.co/klue/roberta-large?text=%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98+%EC%88%98%EB%8F%84%EB%8A%94+%5BMASK%5D+%EC%9E%85%EB%8B%88%EB%8B%A4.
    txt_model_name = "klue/roberta-large"

    #### read train and test csv data
    all_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)


    ### label preprocessing: korean label to integer label
    le_1 = preprocessing.LabelEncoder()
    le_1.fit(all_df['cat1'].values)
    all_df['cat1'] = le_1.transform(all_df['cat1'].values)

    le_2 = preprocessing.LabelEncoder()
    le_2.fit(all_df['cat2'].values)
    all_df['cat2'] = le_2.transform(all_df['cat2'].values)

    le_3 = preprocessing.LabelEncoder()
    le_3.fit(all_df['cat3'].values)
    all_df['cat3'] = le_3.transform(all_df['cat3'].values)

    #### pre-defined numpy array for storing predictions for each fold
    pred_probs_1_folds = np.zeros([len(test_df), 6])
    pred_probs_2_folds = np.zeros([len(test_df), 18])
    pred_probs_3_folds = np.zeros([len(test_df), 128])
    pred_probs_ens_folds = np.zeros([len(test_df), 128])

    #### stratified 5 fold train data split by cat3 label
    #### the output of this script is ensemble of 5 folds
    skf = StratifiedKFold(n_splits=5)
    for fold_i, (train_index, valid_index) in enumerate(skf.split(all_df, all_df['cat3'])):
        print(f"Fold: {fold_i}")
        train_df = all_df.iloc[train_index]
        val_df = all_df.iloc[valid_index]

        cat1_tree_dict, cat2_tree_dict = get_label_trees(all_df)

        ## model dataset dataloader optimizer scheduler configure
        tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
        config = AutoConfig.from_pretrained(txt_model_name)
        txt_model = AutoModelForSequenceClassification.from_pretrained(txt_model_name, config=config, ignore_mismatched_sizes=True)

        train_dataset = CustomDataset(train_df, infer=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        distill_dataset = CustomDataset_Distill(test_df, distill_cat1, distill_cat2, distill_cat3)
        distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

        val_dataset = CustomDataset(val_df, infer=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = CustomModel(txt_model, tokenizer, max_length=max_length,
                            cat1_classes=len(le_1.classes_), cat2_classes=len(le_2.classes_),
                            cat3_classes=len(le_3.classes_))
        model.to(DEVICE)

        optimizer = torch.optim.RAdam(params=model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        infer_model = train(model, optimizer, train_loader, distill_loader, val_loader, scheduler, cat1_tree_dict,
                            cat2_tree_dict,
                            model_save_dir, max_epoch=max_epoch, fold_i=fold_i)

        ## test inference
        test_dataset = CustomDataset(test_df, infer=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model_preds, pred_probs_ens, pred_probs_1, pred_probs_2, pred_probs_3 = inference(infer_model, test_loader,
                                                                                          cat1_tree_dict,
                                                                                          cat2_tree_dict)

        ### ensemble 5 fold results, probability
        pred_probs_1_folds += pred_probs_1
        pred_probs_2_folds += pred_probs_2
        pred_probs_3_folds += pred_probs_3
        pred_probs_ens_folds += pred_probs_ens


    #### save final submission for distillation learning
    preds_ens_folds = np.argmax(pred_probs_ens_folds, axis=1)
    submit = pd.read_csv(sample_submit_csv_path)
    submit['cat3'] = le_3.inverse_transform(preds_ens_folds)
    submit.to_csv(test_submit_csv_path, index=False)


if __name__ == '__main__':
    main()







