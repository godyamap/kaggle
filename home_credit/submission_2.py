# %% [code]
# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.027198Z","iopub.execute_input":"2024-02-09T09:48:03.027837Z","iopub.status.idle":"2024-02-09T09:48:03.033926Z","shell.execute_reply.started":"2024-02-09T09:48:03.027805Z","shell.execute_reply":"2024-02-09T09:48:03.032966Z"}}
import gc
import warnings
from glob import glob
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedGroupKFold

warnings.simplefilter(action="ignore", category=FutureWarning)

# %% [markdown]
# ### Pre-Fitted Voting Model


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.070798Z","iopub.execute_input":"2024-02-09T09:48:03.071062Z","iopub.status.idle":"2024-02-09T09:48:03.077777Z","shell.execute_reply.started":"2024-02-09T09:48:03.07104Z","shell.execute_reply":"2024-02-09T09:48:03.076775Z"}}
class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


# %% [markdown]
# ### Pipeline


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.103105Z","iopub.execute_input":"2024-02-09T09:48:03.10338Z","iopub.status.idle":"2024-02-09T09:48:03.115494Z","shell.execute_reply.started":"2024-02-09T09:48:03.103358Z","shell.execute_reply":"2024-02-09T09:48:03.114577Z"}}
class Pipeline:
    @staticmethod
    def set_table_dtypes(df):  # dtype繧堤ｵｱ荳縺吶ｋ
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))

        return df

    @staticmethod
    def handle_dates(df):  # D縺ｮ迚ｹ蠕ｴ驥上ｒdate_decision縺九ｉ縺ｮ蟾ｮ蛻�律謨ｰ縺ｫ螟画峩縺吶ｋ
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())

        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(
        df,
    ):  # 迚ｹ蠕ｴ驥上↓縺､縺�※縲］ull縺�95%莉･荳翫�繧ゅ�縺ｨnunique縺�1<nunique<200縺ｫ隧ｲ蠖薙＠縺ｪ縺�ｂ縺ｮ繧貞炎髯､縺吶ｋ
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (
                df[col].dtype == pl.String
            ):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


# %% [markdown]
# ### Automatic Aggregation


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.125752Z","iopub.execute_input":"2024-02-09T09:48:03.126041Z","iopub.status.idle":"2024-02-09T09:48:03.142071Z","shell.execute_reply.started":"2024-02-09T09:48:03.126018Z","shell.execute_reply":"2024-02-09T09:48:03.141095Z"}}
class Aggregator:
    @staticmethod
    def num_expr(df):  # P縺ｨA縺ｮ迚ｹ蠕ｴ驥上↓縺､縺�※縲∵怙螟ｧ蛟､繧貞叙繧雁�縺励※迚ｹ蠕ｴ驥上↓霑ｽ蜉�
        cols = [col for col in df.columns if col[-1] in ("P", "A")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def date_expr(df):  # D縺ｮ迚ｹ蠕ｴ驥上↓縺､縺�※縲∵怙螟ｧ蛟､繧貞叙繧雁�縺励※迚ｹ蠕ｴ驥上↓霑ｽ蜉�
        cols = [col for col in df.columns if col[-1] in ("D",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def str_expr(df):  # M縺ｮ迚ｹ蠕ｴ驥上↓縺､縺�※縲∵怙螟ｧ蛟､繧貞叙繧雁�縺励※迚ｹ蠕ｴ驥上↓霑ｽ蜉�
        cols = [col for col in df.columns if col[-1] in ("M",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def other_expr(df):  # T縺ｨL縺ｮ迚ｹ蠕ｴ驥上↓縺､縺�※縲∵怙螟ｧ蛟､繧貞叙繧雁�縺励※迚ｹ蠕ｴ驥上↓霑ｽ蜉�
        cols = [col for col in df.columns if col[-1] in ("T", "L")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def count_expr(df):  # num_group縺ｫ縺､縺�※縲∵怙螟ｧ蛟､繧貞叙繧雁�縺励※迚ｹ蠕ｴ驥上↓霑ｽ蜉�
        cols = [col for col in df.columns if "num_group" in col]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def get_exprs(df):  # 荳願ｨ倥�髢｢謨ｰ繧貞ｮ溯｡後＠縺ｦ縲∫ｵ先棡繧定ｿ斐☆
        maxexprs = (
            Aggregator.num_expr(df)[0]
            + Aggregator.date_expr(df)[0]
            + Aggregator.str_expr(df)[0]
            + Aggregator.other_expr(df)[0]
            + Aggregator.count_expr(df)[0]
        )

        minexprs = (
            Aggregator.num_expr(df)[1]
            + Aggregator.date_expr(df)[1]
            + Aggregator.str_expr(df)[1]
            + Aggregator.other_expr(df)[1]
            + Aggregator.count_expr(df)[1]
        )

        return maxexprs, minexprs


# %% [markdown]
# ### File I/O


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.152047Z","iopub.execute_input":"2024-02-09T09:48:03.152396Z","iopub.status.idle":"2024-02-09T09:48:03.15998Z","shell.execute_reply.started":"2024-02-09T09:48:03.152372Z","shell.execute_reply":"2024-02-09T09:48:03.15894Z"}}
def read_file(
    path, depth=None
):  # dtype縺ｮ邨ｱ荳縲…ase_id縺ｧ繧ｰ繝ｫ繝ｼ繝怜喧縺励◆荳翫〒縺ｮ譛螟ｧ迚ｹ蠕ｴ驥上�霑ｽ蜉�
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)

    if depth in [1, 2]:
        maxexprs, minexprs = Aggregator.get_exprs(df)
        df = df.group_by("case_id").agg(*maxexprs, *minexprs)

    return df


def read_files(
    regex_path, depth=None
):  # dtype縺ｮ邨ｱ荳縲∬､�焚df縺ｮ邵ｦ譁ｹ蜷代〒縺ｮ邨仙粋縲…ase_id縺ｧ繧ｰ繝ｫ繝ｼ繝怜喧縺励◆荳翫〒縺ｮ譛螟ｧ迚ｹ蠕ｴ驥上�霑ｽ蜉�
    chunks = []
    for path in glob(str(regex_path)):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))

    df = pl.concat(chunks, how="vertical_relaxed")
    if depth in [1, 2]:
        maxexprs, minexprs = Aggregator.get_exprs(df)
        df = df.group_by("case_id").agg(*maxexprs, *minexprs)

    return df


# %% [markdown]
# ### Feature Engineering


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.17866Z","iopub.execute_input":"2024-02-09T09:48:03.179262Z","iopub.status.idle":"2024-02-09T09:48:03.184547Z","shell.execute_reply.started":"2024-02-09T09:48:03.179239Z","shell.execute_reply":"2024-02-09T09:48:03.183747Z"}}
def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )

    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    df_base = df_base.pipe(Pipeline.handle_dates)

    return df_base


# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.200261Z","iopub.execute_input":"2024-02-09T09:48:03.200736Z","iopub.status.idle":"2024-02-09T09:48:03.205449Z","shell.execute_reply.started":"2024-02-09T09:48:03.200712Z","shell.execute_reply":"2024-02-09T09:48:03.204594Z"}}
def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()

    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)

    df_data[cat_cols] = df_data[cat_cols].astype("category")

    return df_data, cat_cols


# %% [markdown]
# ### Configuration

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.22407Z","iopub.execute_input":"2024-02-09T09:48:03.224516Z","iopub.status.idle":"2024-02-09T09:48:03.228747Z","shell.execute_reply.started":"2024-02-09T09:48:03.224491Z","shell.execute_reply":"2024-02-09T09:48:03.227697Z"}}
ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"

# %% [markdown]
# ### Train Files Read & Feature Engineering

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:03.255415Z","iopub.execute_input":"2024-02-09T09:48:03.255683Z","iopub.status.idle":"2024-02-09T09:48:36.031378Z","shell.execute_reply.started":"2024-02-09T09:48:03.255661Z","shell.execute_reply":"2024-02-09T09:48:36.030568Z"}}
data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
    ],
}

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:36.032914Z","iopub.execute_input":"2024-02-09T09:48:36.033199Z","iopub.status.idle":"2024-02-09T09:48:46.392892Z","shell.execute_reply.started":"2024-02-09T09:48:36.033176Z","shell.execute_reply":"2024-02-09T09:48:46.391889Z"}}
df_train = feature_eng(**data_store)

print("train data shape:\t", df_train.shape)

# %% [markdown]
# ### Test Files Read & Feature Engineering

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:46.393984Z","iopub.execute_input":"2024-02-09T09:48:46.394256Z","iopub.status.idle":"2024-02-09T09:48:46.841917Z","shell.execute_reply.started":"2024-02-09T09:48:46.394233Z","shell.execute_reply":"2024-02-09T09:48:46.840869Z"}}
data_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        read_file(TEST_DIR / "test_other_1.parquet", 1),
        read_file(TEST_DIR / "test_person_1.parquet", 1),
        read_file(TEST_DIR / "test_deposit_1.parquet", 1),
        read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
    ],
}

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:46.843923Z","iopub.execute_input":"2024-02-09T09:48:46.844216Z","iopub.status.idle":"2024-02-09T09:48:46.882465Z","shell.execute_reply.started":"2024-02-09T09:48:46.844193Z","shell.execute_reply":"2024-02-09T09:48:46.881617Z"}}
df_test = feature_eng(**data_store)

print("test data shape:\t", df_test.shape)

# %% [markdown]
# ### Feature Elimination

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:46.883433Z","iopub.execute_input":"2024-02-09T09:48:46.883703Z","iopub.status.idle":"2024-02-09T09:48:50.210425Z","shell.execute_reply.started":"2024-02-09T09:48:46.88368Z","shell.execute_reply":"2024-02-09T09:48:50.209573Z"}}
df_train = df_train.pipe(Pipeline.filter_cols)
df_test = df_test.select([col for col in df_train.columns if col != "target"])

print("train data shape:\t", df_train.shape)
print("test data shape:\t", df_test.shape)

# %% [markdown]
# ### Pandas Conversion

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:48:50.211672Z","iopub.execute_input":"2024-02-09T09:48:50.211958Z","iopub.status.idle":"2024-02-09T09:49:10.31903Z","shell.execute_reply.started":"2024-02-09T09:48:50.211934Z","shell.execute_reply":"2024-02-09T09:49:10.318243Z"}}
df_train, cat_cols = to_pandas(df_train)
df_test, cat_cols = to_pandas(df_test, cat_cols)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:10.320134Z","iopub.execute_input":"2024-02-09T09:49:10.320467Z","iopub.status.idle":"2024-02-09T09:49:10.350379Z","shell.execute_reply.started":"2024-02-09T09:49:10.320437Z","shell.execute_reply":"2024-02-09T09:49:10.348974Z"}}
print(df_train.shape)
df_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:10.351425Z","iopub.execute_input":"2024-02-09T09:49:10.351715Z","iopub.status.idle":"2024-02-09T09:49:10.373798Z","shell.execute_reply.started":"2024-02-09T09:49:10.351692Z","shell.execute_reply":"2024-02-09T09:49:10.372961Z"}}
print(df_test.shape)
df_test.head()

# %% [markdown]
# ### Garbage Collection

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:10.374793Z","iopub.execute_input":"2024-02-09T09:49:10.375026Z","iopub.status.idle":"2024-02-09T09:49:10.520142Z","shell.execute_reply.started":"2024-02-09T09:49:10.375006Z","shell.execute_reply":"2024-02-09T09:49:10.519203Z"}}
del data_store

gc.collect()

# %% [markdown]
# ### EDA

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:10.522892Z","iopub.execute_input":"2024-02-09T09:49:10.523199Z","iopub.status.idle":"2024-02-09T09:49:10.5602Z","shell.execute_reply.started":"2024-02-09T09:49:10.523173Z","shell.execute_reply":"2024-02-09T09:49:10.559314Z"}}
print("Train is duplicated:\t", df_train["case_id"].duplicated().any())
print("Train Week Range:\t", (df_train["WEEK_NUM"].min(), df_train["WEEK_NUM"].max()))

print()

print("Test is duplicated:\t", df_test["case_id"].duplicated().any())
print("Test Week Range:\t", (df_test["WEEK_NUM"].min(), df_test["WEEK_NUM"].max()))

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:10.561309Z","iopub.execute_input":"2024-02-09T09:49:10.561613Z","iopub.status.idle":"2024-02-09T09:49:27.090185Z","shell.execute_reply.started":"2024-02-09T09:49:10.561583Z","shell.execute_reply":"2024-02-09T09:49:27.089229Z"}}
sns.lineplot(
    data=df_train,
    x="WEEK_NUM",
    y="target",
)
plt.show()

# %% [markdown]
# ### Training

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T09:49:27.091366Z","iopub.execute_input":"2024-02-09T09:49:27.091672Z","iopub.status.idle":"2024-02-09T10:05:11.944986Z","shell.execute_reply.started":"2024-02-09T09:49:27.091647Z","shell.execute_reply":"2024-02-09T10:05:11.94387Z"}}
# (update_yamashita)add train_data for predicition
X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
y = df_train["target"]
weeks = df_train["WEEK_NUM"]

cv = StratifiedGroupKFold(n_splits=10, shuffle=False)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "device": "gpu",
}

fitted_models = []

for idx_train, idx_valid in cv.split(X, y, groups=weeks):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
    )

    fitted_models.append(model)

model = VotingModel(fitted_models)

# %% [markdown]
# ### Prediction

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T10:05:11.9465Z","iopub.execute_input":"2024-02-09T10:05:11.947104Z","iopub.status.idle":"2024-02-09T10:05:12.25451Z","shell.execute_reply.started":"2024-02-09T10:05:11.947067Z","shell.execute_reply":"2024-02-09T10:05:12.253712Z"}}
X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = X_test.set_index("case_id")

# (update_yamashita)add prediction of train_data
X_train = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
y_pred_train_2 = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

# %% [markdown]
# ### Submission

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T10:05:12.255651Z","iopub.execute_input":"2024-02-09T10:05:12.255954Z","iopub.status.idle":"2024-02-09T10:05:12.266758Z","shell.execute_reply.started":"2024-02-09T10:05:12.255929Z","shell.execute_reply":"2024-02-09T10:05:12.265783Z"}}
df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T10:05:12.267966Z","iopub.execute_input":"2024-02-09T10:05:12.26829Z","iopub.status.idle":"2024-02-09T10:05:12.280505Z","shell.execute_reply.started":"2024-02-09T10:05:12.268258Z","shell.execute_reply":"2024-02-09T10:05:12.279618Z"}}
print("Check null: ", df_subm["score"].isnull().any())

df_subm.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-09T10:05:12.281649Z","iopub.execute_input":"2024-02-09T10:05:12.281977Z","iopub.status.idle":"2024-02-09T10:05:12.291529Z","shell.execute_reply.started":"2024-02-09T10:05:12.281945Z","shell.execute_reply":"2024-02-09T10:05:12.29079Z"}}
df_subm.to_csv("submission.csv")

# %% [code]