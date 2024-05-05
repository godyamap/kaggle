# %% [code]
# %% [code] {"papermill":{"duration":4.235504,"end_time":"2024-02-06T07:31:41.10751","exception":false,"start_time":"2024-02-06T07:31:36.872006","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:16:57.654702Z","iopub.execute_input":"2024-02-07T01:16:57.655486Z","iopub.status.idle":"2024-02-07T01:17:03.747169Z","shell.execute_reply.started":"2024-02-07T01:16:57.655455Z","shell.execute_reply":"2024-02-07T01:17:03.746098Z"}}
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 

dataPath = "/kaggle/input/home-credit-credit-risk-model-stability/"

# %% [code] {"papermill":{"duration":0.01679,"end_time":"2024-02-06T07:31:41.129274","exception":false,"start_time":"2024-02-06T07:31:41.112484","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:03.749192Z","iopub.execute_input":"2024-02-07T01:17:03.749543Z","iopub.status.idle":"2024-02-07T01:17:03.758437Z","shell.execute_reply.started":"2024-02-07T01:17:03.749512Z","shell.execute_reply":"2024-02-07T01:17:03.75738Z"}}
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

    return df

def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:  
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df

# %% [code] {"papermill":{"duration":12.392163,"end_time":"2024-02-06T07:31:53.526418","exception":false,"start_time":"2024-02-06T07:31:41.134255","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:03.759503Z","iopub.execute_input":"2024-02-07T01:17:03.759825Z","iopub.status.idle":"2024-02-07T01:17:18.832078Z","shell.execute_reply.started":"2024-02-07T01:17:03.759798Z","shell.execute_reply":"2024-02-07T01:17:18.831265Z"}}
train_basetable = pl.read_csv(dataPath + "csv_files/train/train_base.csv")
train_static = pl.concat(
    [
        pl.read_csv(dataPath + "csv_files/train/train_static_0_0.csv").pipe(set_table_dtypes),
        pl.read_csv(dataPath + "csv_files/train/train_static_0_1.csv").pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
train_static_cb = pl.read_csv(dataPath + "csv_files/train/train_static_cb_0.csv").pipe(set_table_dtypes)
train_person_1 = pl.read_csv(dataPath + "csv_files/train/train_person_1.csv").pipe(set_table_dtypes) 
train_credit_bureau_b_2 = pl.read_csv(dataPath + "csv_files/train/train_credit_bureau_b_2.csv").pipe(set_table_dtypes) 

# %% [code] {"papermill":{"duration":0.117972,"end_time":"2024-02-06T07:31:53.649206","exception":false,"start_time":"2024-02-06T07:31:53.531234","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:18.834639Z","iopub.execute_input":"2024-02-07T01:17:18.835224Z","iopub.status.idle":"2024-02-07T01:17:18.915442Z","shell.execute_reply.started":"2024-02-07T01:17:18.835196Z","shell.execute_reply":"2024-02-07T01:17:18.914558Z"}}
test_basetable = pl.read_csv(dataPath + "csv_files/test/test_base.csv")
test_static = pl.concat(
    [
        pl.read_csv(dataPath + "csv_files/test/test_static_0_0.csv").pipe(set_table_dtypes),
        pl.read_csv(dataPath + "csv_files/test/test_static_0_1.csv").pipe(set_table_dtypes),
        pl.read_csv(dataPath + "csv_files/test/test_static_0_2.csv").pipe(set_table_dtypes),
    ],
    how="vertical_relaxed",
)
test_static_cb = pl.read_csv(dataPath + "csv_files/test/test_static_cb_0.csv").pipe(set_table_dtypes)
test_person_1 = pl.read_csv(dataPath + "csv_files/test/test_person_1.csv").pipe(set_table_dtypes) 
test_credit_bureau_b_2 = pl.read_csv(dataPath + "csv_files/test/test_credit_bureau_b_2.csv").pipe(set_table_dtypes) 

# %% [markdown] {"papermill":{"duration":0.004244,"end_time":"2024-02-06T07:31:53.65836","exception":false,"start_time":"2024-02-06T07:31:53.654116","status":"completed"},"tags":[]}
# ## Feature engineering
# 
# In this part, we can see a simple example of joining tables via `case_id`. Here the loading and joining is done with polars library. Polars library is blazingly fast and has much smaller memory footprint than pandas. 

# %% [code] {"papermill":{"duration":1.263082,"end_time":"2024-02-06T07:31:54.925894","exception":false,"start_time":"2024-02-06T07:31:53.662812","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:18.916872Z","iopub.execute_input":"2024-02-07T01:17:18.917155Z","iopub.status.idle":"2024-02-07T01:17:20.913622Z","shell.execute_reply.started":"2024-02-07T01:17:18.917132Z","shell.execute_reply":"2024-02-07T01:17:20.912828Z"}}
# We need to use aggregation functions in tables with depth > 1, so tables that contain num_group1 column or 
# also num_group2 column.
train_person_1_feats_1 = train_person_1.group_by("case_id").agg(
    pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
    (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
)

# Here num_group1=0 has special meaning, it is the person who applied for the loan.
train_person_1_feats_2 = train_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
    pl.col("num_group1") == 0
).drop("num_group1").rename({"housetype_905L": "person_housetype"})

# Here we have num_goup1 and num_group2, so we need to aggregate again.
train_credit_bureau_b_2_feats = train_credit_bureau_b_2.group_by("case_id").agg(
    pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
    (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
)

# We will process in this examples only A-type and M-type columns, so we need to select them.
selected_static_cols = []
for col in train_static.columns:
    if col[-1] in ("A", "M"):
        selected_static_cols.append(col)
print(selected_static_cols)

selected_static_cb_cols = []
for col in train_static_cb.columns:
    if col[-1] in ("A", "M"):
        selected_static_cb_cols.append(col)
print(selected_static_cb_cols)

# Join all tables together.
data = train_basetable.join(
    train_static.select(["case_id"]+selected_static_cols), how="left", on="case_id"
).join(
    train_static_cb.select(["case_id"]+selected_static_cb_cols), how="left", on="case_id"
).join(
    train_person_1_feats_1, how="left", on="case_id"
).join(
    train_person_1_feats_2, how="left", on="case_id"
).join(
    train_credit_bureau_b_2_feats, how="left", on="case_id"
)

# %% [code] {"papermill":{"duration":0.426199,"end_time":"2024-02-06T07:31:55.356927","exception":false,"start_time":"2024-02-06T07:31:54.930728","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:20.914816Z","iopub.execute_input":"2024-02-07T01:17:20.915136Z","iopub.status.idle":"2024-02-07T01:17:21.842495Z","shell.execute_reply.started":"2024-02-07T01:17:20.915091Z","shell.execute_reply":"2024-02-07T01:17:21.841429Z"}}
test_person_1_feats_1 = train_person_1.group_by("case_id").agg(
    pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
    (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
)

test_person_1_feats_2 = train_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
    pl.col("num_group1") == 0
).drop("num_group1").rename({"housetype_905L": "person_housetype"})

test_credit_bureau_b_2_feats = train_credit_bureau_b_2.group_by("case_id").agg(
    pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
    (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
)

data_submission = test_basetable.join(
    test_static.select(["case_id"]+selected_static_cols), how="left", on="case_id"
).join(
    test_static_cb.select(["case_id"]+selected_static_cb_cols), how="left", on="case_id"
).join(
    test_person_1_feats_1, how="left", on="case_id"
).join(
    test_person_1_feats_2, how="left", on="case_id"
).join(
    test_credit_bureau_b_2_feats, how="left", on="case_id"
)

# %% [code] {"papermill":{"duration":4.984828,"end_time":"2024-02-06T07:32:00.346668","exception":false,"start_time":"2024-02-06T07:31:55.36184","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:21.843646Z","iopub.execute_input":"2024-02-07T01:17:21.843928Z","iopub.status.idle":"2024-02-07T01:17:29.219417Z","shell.execute_reply.started":"2024-02-07T01:17:21.843906Z","shell.execute_reply":"2024-02-07T01:17:29.218184Z"}}
case_ids = data["case_id"].unique().shuffle(seed=1)
case_ids_train, case_ids_test = train_test_split(case_ids, train_size=0.6, random_state=1)
case_ids_valid, case_ids_test = train_test_split(case_ids_test, train_size=0.5, random_state=1)

cols_pred = []
for col in data.columns:
    if col[-1].isupper() and col[:-1].islower():
        cols_pred.append(col)

print(cols_pred)

def from_polars_to_pandas(case_ids: pl.DataFrame) -> pl.DataFrame:
    return (
        data.filter(pl.col("case_id").is_in(case_ids))[["case_id", "WEEK_NUM", "target"]].to_pandas(),
        data.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas(),
        data.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    )

base_train, X_train, y_train = from_polars_to_pandas(case_ids_train)
base_valid, X_valid, y_valid = from_polars_to_pandas(case_ids_valid)
base_test, X_test, y_test = from_polars_to_pandas(case_ids_test)

for df in [X_train, X_valid, X_test]:
    df = convert_strings(df)

# %% [code] {"papermill":{"duration":0.013708,"end_time":"2024-02-06T07:32:00.365522","exception":false,"start_time":"2024-02-06T07:32:00.351814","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:29.220704Z","iopub.execute_input":"2024-02-07T01:17:29.220991Z","iopub.status.idle":"2024-02-07T01:17:29.226011Z","shell.execute_reply.started":"2024-02-07T01:17:29.220968Z","shell.execute_reply":"2024-02-07T01:17:29.224999Z"}}
print(f"Train: {X_train.shape}")
print(f"Valid: {X_valid.shape}")
print(f"Test: {X_test.shape}")

# %% [markdown] {"papermill":{"duration":0.004343,"end_time":"2024-02-06T07:32:00.374801","exception":false,"start_time":"2024-02-06T07:32:00.370458","status":"completed"},"tags":[]}
# # **Training**

# %% [code] {"papermill":{"duration":94.027767,"end_time":"2024-02-06T07:33:34.407185","exception":false,"start_time":"2024-02-06T07:32:00.379418","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:29.227137Z","iopub.execute_input":"2024-02-07T01:17:29.227433Z","iopub.status.idle":"2024-02-07T01:17:46.730937Z","shell.execute_reply.started":"2024-02-07T01:17:29.22741Z","shell.execute_reply":"2024-02-07T01:17:46.730157Z"}}

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 5,
    "num_leaves": 50, 
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 3000,
    "verbose": -1,
    'device':'gpu'
}


# StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#List for saving models
lgb_models = []

# StratifiedKFold loop
for train_index, valid_index in skf.split(X_train, y_train):
    X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=lgb_train)

    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid],callbacks=[lgb.log_evaluation(50), lgb.early_stopping(10)])

    # Save model
    lgb_models.append(gbm)


# %% [code] {"execution":{"iopub.status.busy":"2024-02-07T01:17:46.736068Z","iopub.execute_input":"2024-02-07T01:17:46.739193Z","iopub.status.idle":"2024-02-07T01:17:46.751499Z","shell.execute_reply.started":"2024-02-07T01:17:46.739102Z","shell.execute_reply":"2024-02-07T01:17:46.749696Z"}}
lgb_models

# %% [code] {"papermill":{"duration":2476.526221,"end_time":"2024-02-06T08:14:50.939246","exception":false,"start_time":"2024-02-06T07:33:34.413025","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:46.753592Z","iopub.execute_input":"2024-02-07T01:17:46.754472Z","iopub.status.idle":"2024-02-07T01:17:52.237159Z","shell.execute_reply.started":"2024-02-07T01:17:46.754414Z","shell.execute_reply":"2024-02-07T01:17:52.236102Z"}}
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    device="cuda",
    objective='binary:logistic',
    tree_method="hist",
    enable_categorical=True,
    eval_metric='auc',
    subsample=1,
    colsample_bytree=1,
    min_child_weight=1,
    max_depth=20,
    #gamma=0.7,
    #reg_alpha=0.7,
    n_estimators=1200,
    random_state=42,
)

# Training the model on the training data
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=100,
    verbose=True,
)

# %% [code] {"papermill":{"duration":2283.103416,"end_time":"2024-02-06T08:52:54.076015","exception":false,"start_time":"2024-02-06T08:14:50.972599","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:17:52.238383Z","iopub.execute_input":"2024-02-07T01:17:52.239026Z","iopub.status.idle":"2024-02-07T01:18:30.093513Z","shell.execute_reply.started":"2024-02-07T01:17:52.238991Z","shell.execute_reply":"2024-02-07T01:18:30.092424Z"}}
import pandas as pd
from catboost import CatBoostClassifier

# 找出分类特征
cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category' or X_train[col].dtype.name == 'object']

# 为每个分类特征添加新类别 'Missing' 并替换 NaN 值
for col in cat_features:
    X_train[col] = X_train[col].cat.add_categories('Missing').fillna('Missing')
    X_valid[col] = X_valid[col].cat.add_categories('Missing').fillna('Missing')

cat_model = CatBoostClassifier(
    iterations=1200,                 
    depth=12,                        
    learning_rate=0.1,               
    eval_metric='AUC',               
    random_seed=42,                  
    bootstrap_type='Bayesian',       
    bagging_temperature=1,           
    od_type='Iter',                  
    od_wait=50,
    task_type='GPU'
)

# 训练模型
cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features,  # 明确指定了分类特征
    use_best_model=True,
    verbose=True
)


# %% [markdown] {"papermill":{"duration":0.064039,"end_time":"2024-02-06T08:52:54.203289","exception":false,"start_time":"2024-02-06T08:52:54.13925","status":"completed"},"tags":[]}
# Evaluation with AUC and then comparison with the stability metric is shown below.

# %% [code] {"papermill":{"duration":0.078345,"end_time":"2024-02-06T08:52:54.345833","exception":false,"start_time":"2024-02-06T08:52:54.267488","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:18:30.094747Z","iopub.execute_input":"2024-02-07T01:18:30.095553Z","iopub.status.idle":"2024-02-07T01:18:30.100542Z","shell.execute_reply.started":"2024-02-07T01:18:30.095521Z","shell.execute_reply":"2024-02-07T01:18:30.099617Z"}}
# from sklearn.metrics import roc_auc_score

# for model, name in [(lgb_model, "lgb"), (xgb_model, "xgb")]:
#     for base, X in [(base_train, X_train), (base_valid, X_valid), (base_test, X_test)]:
#         y_pred = model.predict(X, num_iteration=model.best_iteration) if name == "lgb" else model.predict(X)
#         base[f"{name}_score"] = y_pred
        
# for base, X in [(base_train, X_train), (base_valid, X_valid), (base_test, X_test)]:

#     X_processed = X.copy()
#     cat_features = [col for col in X_processed.columns if X_processed[col].dtype.name == 'category' or X_processed[col].dtype.name == 'object']
#     for col in cat_features:
#         X_processed[col] = X_processed[col].cat.add_categories('Missing').fillna('Missing')

#     y_pred = cat_model.predict(X_processed)
#     base["cat_score"] = y_pred

# for base in [base_train, base_valid, base_test]:
#     base['combined_score'] = base[['lgb_score', 'xgb_score', 'cat_score']].mean(axis=1)
#     print(f'The AUC score of combined models on the {base.name} set is: {roc_auc_score(base["target"], base["combined_score"])}')
 

# %% [code] {"papermill":{"duration":0.072886,"end_time":"2024-02-06T08:52:54.48175","exception":false,"start_time":"2024-02-06T08:52:54.408864","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:18:30.10173Z","iopub.execute_input":"2024-02-07T01:18:30.101999Z","iopub.status.idle":"2024-02-07T01:18:30.114241Z","shell.execute_reply.started":"2024-02-07T01:18:30.101977Z","shell.execute_reply":"2024-02-07T01:18:30.113383Z"}}
# def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
#     gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
#         .sort_values("WEEK_NUM")\
#         .groupby("WEEK_NUM")[["target", "score"]]\
#         .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()
    
#     x = np.arange(len(gini_in_time))
#     y = gini_in_time
#     a, b = np.polyfit(x, y, 1)
#     y_hat = a*x + b
#     residuals = y - y_hat
#     res_std = np.std(residuals)
#     avg_gini = np.mean(gini_in_time)
#     return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

# model_names = ["lgb", "xgb", "cat", "combined"]
# for model_name in model_names:
#     score_column = f"{model_name}_score" if model_name != "combined" else "combined_score"
#     stability_score_train = gini_stability(base_train, score_column)
#     stability_score_valid = gini_stability(base_valid, score_column)
#     stability_score_test = gini_stability(base_test, score_column)

#     print(f'The stability score of {model_name} on the train set is: {stability_score_train}') 
#     print(f'The stability score of {model_name} on the valid set is: {stability_score_valid}') 
#     print(f'The stability score of {model_name} on the test set is: {stability_score_test}')


# %% [markdown] {"papermill":{"duration":0.062845,"end_time":"2024-02-06T08:52:54.607523","exception":false,"start_time":"2024-02-06T08:52:54.544678","status":"completed"},"tags":[]}
# ## Submission
# 
# Scoring the submission dataset is below, we need to take care of new categories. Then we save the score as a last step. 

# %% [code] {"papermill":{"duration":0.204329,"end_time":"2024-02-06T08:52:54.873928","exception":false,"start_time":"2024-02-06T08:52:54.669599","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:18:30.115381Z","iopub.execute_input":"2024-02-07T01:18:30.115943Z","iopub.status.idle":"2024-02-07T01:18:30.346644Z","shell.execute_reply.started":"2024-02-07T01:18:30.115918Z","shell.execute_reply":"2024-02-07T01:18:30.344387Z"}}
X_submission = data_submission[cols_pred].to_pandas()
X_submission = convert_strings(X_submission)
X_submission_processed = X_submission.copy()
categorical_cols = X_train.select_dtypes(include=['category']).columns

for col in categorical_cols:
    train_categories = set(X_train[col].cat.categories)
    submission_categories = set(X_submission[col].cat.categories)
    new_categories = submission_categories - train_categories
    X_submission.loc[X_submission[col].isin(new_categories), col] = "Unknown"
    new_dtype = pd.CategoricalDtype(categories=train_categories, ordered=True)
    X_train[col] = X_train[col].astype(new_dtype)
    X_submission[col] = X_submission[col].astype(new_dtype)

for col in cat_features:
    X_submission_processed[col] = X_submission_processed[col].cat.add_categories('Missing').fillna('Missing')

#Make prediction
def predict_with_fold_models(models, X):
    predictions = [model.predict(X, num_iteration=model.best_iteration) for model in models]
    return np.mean(predictions, axis=0)

lgb_pred = predict_with_fold_models(lgb_models, X_submission) 

xgb_pred = xgb_model.predict(X_submission)

cat_pred = cat_model.predict(X_submission_processed)

weight_lgb = 0.5
weight_xgb = 0.25 
weight_cat = 0.25

assert weight_lgb + weight_xgb + weight_cat == 1, "The sum of weights must be 1."

y_submission_pred = (weight_lgb * lgb_pred) + (weight_xgb * xgb_pred) + (weight_cat * cat_pred)




# %% [code] {"papermill":{"duration":0.089784,"end_time":"2024-02-06T08:52:55.026753","exception":false,"start_time":"2024-02-06T08:52:54.936969","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:18:30.34765Z","iopub.execute_input":"2024-02-07T01:18:30.348282Z","iopub.status.idle":"2024-02-07T01:18:30.357145Z","shell.execute_reply.started":"2024-02-07T01:18:30.348256Z","shell.execute_reply":"2024-02-07T01:18:30.356183Z"}}
submission = pd.DataFrame({
    "case_id": data_submission["case_id"].to_numpy(),
    "score": y_submission_pred
}).set_index('case_id')
submission.to_csv("./submission.csv")

# %% [code] {"papermill":{"duration":0.086803,"end_time":"2024-02-06T08:52:55.176296","exception":false,"start_time":"2024-02-06T08:52:55.089493","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-07T01:18:30.358299Z","iopub.execute_input":"2024-02-07T01:18:30.358582Z","iopub.status.idle":"2024-02-07T01:18:30.372662Z","shell.execute_reply.started":"2024-02-07T01:18:30.358559Z","shell.execute_reply":"2024-02-07T01:18:30.371674Z"}}
submission