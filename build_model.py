import pandas as pd
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# load in correct answers
df_ans = pd.read_json("./datasets/data_mc.json", orient='records', lines=True)

# load in model predictions
df_dict_full = defaultdict(pd.DataFrame)

for file in os.listdir("results/eval3/"):
    if file.endswith(".json"):
        df = pd.read_json(os.path.join("./results/eval3/", file), orient='index') # transpose the dataframe
        df_dict_full[file.strip(".json")] = df

# use df_dict as subset of df_dict_full, excluding vanilla_t0.9 and dola_static24_t1.0, as well as any file that has top in it
df_dict = {k: v for k, v in df_dict_full.items() if 'top' not in k and 'vanilla_t0.9' not in k and 'dola_static24_t1.0' not in k}

# build dataframe of target: for each prompt in df_ans (each row), the probability of the correct answer (df_ans['answer'])
df_target_prob = pd.DataFrame()
df_target_penalty = pd.DataFrame()
df_target_rank = pd.DataFrame()

for i, row in df_ans.iterrows():
    # get the probability of the correct answer for each decoding strategy
    for x, df in df_dict.items():
        df_target_prob.loc[i, x] = df.loc[i, row['Answer']]

        # mask of incorrect answers
        df_target_penalty.loc[i, x] = (df.loc[i, :].idxmax() == row['Answer'])
        df_target_rank.loc[i, x] = df.loc[i, :].rank(method='min', ascending=False)[row['Answer']]


# target is one of the following: 

# 1) Decoding strategy that maximizes probability of correct answer 
theta = 0.0
y_prob = (df_target_prob - theta * df_target_penalty).idxmax(axis=1)
# 2) Decoding strategy that maximize probability of correct answer with penalty for incorrect answers
theta = 0.1
y_prob_penalty = (df_target_prob - theta * df_target_penalty).idxmax(axis=1)

# 3) Rank of correct answer, default to dola_all_t1.0
df_copy = df_target_rank.copy()
df_copy['dola_all_t1.0'] -= 1e-10
y_rank = df_copy.idxmin(axis=1) # min = better

print(df_dict.keys())

# load in prompt data
df_train = pd.DataFrame(np.load("./datasets/embed_e5-large-v2.npy"), columns=[f"embed_{i}" for i in range(np.load("./datasets/embed_e5-large-v2.npy").shape[1])])
print("Prompt embedding shape:", df_train.shape)

# add in engineered tabular features
df_train['length'] = df_ans['Question'].str.len()
df_train['num_words'] = df_ans['Question'].str.split().str.len()
df_train['num_numbers'] = df_ans['Question'].str.count(r'\d+')
df_train['num_capital_letters'] = df_ans['Question'].str.count(r'[A-Z]')

print("Final feature shape:", df_train.shape)

from sklearn.model_selection import train_test_split

# split indices into train and test
train_idx, test_idx = train_test_split(np.arange(len(df_ans)), test_size=0.3, random_state=42, stratify=df_ans['Type'])

y_bin = y_prob_penalty.apply(lambda x: 1 if 'dola' not in x else 0)
# y_target = y_prob_penalty.astype('category')
y_target = y_rank

# dicts to map decoding strategies to integers and vice versa
dec_dict = {k: i for i, k in enumerate(y_target.unique().sort_values())}
dec_dict_rev = {v: k for k, v in dec_dict.items()}
y_target = y_target.map(dec_dict).astype('int')

X_train = df_train.iloc[train_idx]
y_train = y_target.iloc[train_idx]
df_ans_train = df_ans.iloc[train_idx]

X_test = df_train.iloc[test_idx]
y_test = y_target.iloc[test_idx]
df_ans_test = df_ans.iloc[test_idx]




from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common import space

hyperparameters = {
    # Logistic Regression (Linear Model in AutoGluon)
    'LR': {},
    
    # XGBoost
    'XGB': {
        'ag_args_ensemble': {'fold_fitting_strategy': 'sequential_local'},
        'n_estimators': space.Int(100, 300),
        'max_depth': space.Int(3, 5),
        'learning_rate': space.Real(0.01, 0.1),
    },
    
    # LightGBM
    'GBM': {
        'ag_args_ensemble': {'fold_fitting_strategy': 'sequential_local'},
        'num_boost_round': space.Int(100, 300),
        'num_leaves': space.Int(31, 127),
        'learning_rate': space.Real(0.01, 0.1),
    },
    
    # Random Forest
    'RF': {
        'n_estimators': space.Int(100, 300),
        'max_depth': space.Int(None, 5),
        'max_features': space.Categorical(['sqrt', 'log2', None]),
    },
    
    # TabPFN (if installed)
    'TABPFNV2': {},
    'MITRA': {
        'fine_tune': True,
        'fine_tune_steps': 10
    }
}

hyperparameter_tune_kwargs = {
    'num_trials': 10,
    'scheduler': 'local',
    'searcher': 'auto',
    'time_limit': 3600,
    'num_gpus': 1,
}

label = 'best_dec'
y_train.name = label
y_test.name = label
train_data = TabularDataset(pd.concat([X_train, y_train], axis=1))
test_data = TabularDataset(pd.concat([X_test, y_test], axis=1))

predictor = TabularPredictor(
    label=label, 
    verbosity=1, 
    eval_metric='roc_auc_ovo',
    path='./results/autogluon_models/',

).fit(
    train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    presets='medium_quality',
    num_gpus=1,
    memory_limit=16,
    dynamic_stacking=False
)


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


y_pred_proba = predictor.predict_proba(X_test)
y_pred = y_pred_proba.values.argmax(axis=1)
predictor.leaderboard(test_data, silent=True)

# evaluate model on prompts (how much better is the dynamic strategy than the static strategy?)
acc_on_prompts = {t: {} for t in df_ans_test['Type'].unique()}

# first, get accuracy of plain decoding strategies on prompts
for x, df in df_dict_full.items():
    print("accuracy of", x, ":")
    y_pred_dec = df.iloc[test_idx, :].idxmax(axis=1)
    print(f" - Total:{accuracy_score(df_ans_test['Answer'], y_pred_dec)}")
    print(" - By question type:")
    for t in df_ans_test['Type'].unique():
        acc_on_t = accuracy_score(df_ans_test[df_ans_test['Type'] == t]['Answer'], y_pred_dec[df_ans_test['Type'] == t])
        acc_on_prompts[t][x] = acc_on_t
        print(f"   - {t}: {acc_on_t}")

print("--------------------------------")
# then, get accuracy of model on prompts
y_pred_model = []
for i, (idx, row) in enumerate(df_ans_test.iterrows()):
    # randomly choose a dola decoding strategy
    dec = dec_dict_rev[y_pred[i]]
    y_pred_model.append(df_dict[dec].iloc[idx, :].idxmax(axis=0))

print(f"Accuracy of model: {accuracy_score(df_ans_test['Answer'], pd.Series(y_pred_model))}")
print(" - By question type:")
for t in df_ans_test['Type'].unique():
    acc_on_t = accuracy_score(df_ans_test[df_ans_test['Type'] == t]['Answer'], np.array(y_pred_model)[df_ans_test['Type'] == t])
    acc_on_prompts[t]['model'] = acc_on_t
    print(f"   - {t}: {acc_on_t}")

print(pd.DataFrame(y_pred).map(dec_dict_rev).value_counts())
