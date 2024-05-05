# %% [markdown]
# # Reference
# - [1] [home-credit-baseline](https://www.kaggle.com/code/greysky/home-credit-baseline)
# - [2] [home-credit-baseline-max-min-features](https://www.kaggle.com/code/stechparme/home-credit-baseline-max-min-features)
# - [3] [dependency of autogluon (version confliction by ray package)](https://github.com/autogluon/autogluon/issues/3365)
# - [4] [Autogluon APIs](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)

# # %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-02-10T02:45:23.557162Z","iopub.execute_input":"2024-02-10T02:45:23.557504Z","iopub.status.idle":"2024-02-10T02:45:47.413693Z","shell.execute_reply.started":"2024-02-10T02:45:23.557452Z","shell.execute_reply":"2024-02-10T02:45:47.412518Z"}}
# !python -m pip install --no-index --find-links=/kaggle/input/autogluon-pkgs autogluon > /dev/null

# # %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-02-10T02:45:47.419786Z","iopub.execute_input":"2024-02-10T02:45:47.420055Z","iopub.status.idle":"2024-02-10T02:46:10.396589Z","shell.execute_reply.started":"2024-02-10T02:45:47.420028Z","shell.execute_reply":"2024-02-10T02:46:10.395571Z"}}
# !python -m pip install --no-index --find-links=/kaggle/input/ray-pkgs --upgrade --force-reinstall -q ray==2.6.3

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:10.397908Z","iopub.execute_input":"2024-02-10T02:46:10.398209Z","iopub.status.idle":"2024-02-10T02:46:13.905044Z","shell.execute_reply.started":"2024-02-10T02:46:10.39818Z","shell.execute_reply":"2024-02-10T02:46:13.904277Z"}}
import gc
import warnings
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

warnings.simplefilter(action="ignore", category=FutureWarning)

from autogluon.tabular import TabularDataset, TabularPredictor

# %% [markdown]
# # Pipeline


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:13.907363Z","iopub.execute_input":"2024-02-10T02:46:13.909018Z","iopub.status.idle":"2024-02-10T02:46:14.045002Z","shell.execute_reply.started":"2024-02-10T02:46:13.908989Z","shell.execute_reply":"2024-02-10T02:46:14.043886Z"}}
class Pipeline:
    @staticmethod
    def set_table_dtypes(df):  # Standardize the dtype.
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
    def handle_dates(
        df,
    ):  # Change the feature for D to the difference in days from date_decision.
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())

        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(
        df,
    ):  # Remove those with an average is_null exceeding 0.95 and those that do not fall within the range 1 < nunique < 200.
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
# # Automatic Aggregation


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.046322Z","iopub.execute_input":"2024-02-10T02:46:14.046702Z","iopub.status.idle":"2024-02-10T02:46:14.766971Z","shell.execute_reply.started":"2024-02-10T02:46:14.046666Z","shell.execute_reply":"2024-02-10T02:46:14.765865Z"}}
class Aggregator:
    @staticmethod
    def num_expr(
        df,
    ):  # Extract the maximum and minimum values for features P and A, and add them as additional features.
        cols = [col for col in df.columns if col[-1] in ("P", "A")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def date_expr(
        df,
    ):  # Extract the maximum and minimum values for features D, and add them as additional features.
        cols = [col for col in df.columns if col[-1] in ("D",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def str_expr(
        df,
    ):  # Extract the maximum and minimum values for features M, and add them as additional features.
        cols = [col for col in df.columns if col[-1] in ("M",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def other_expr(
        df,
    ):  # Extract the maximum and minimum values for features T and L, and add them as additional features.
        cols = [col for col in df.columns if col[-1] in ("T", "L")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def count_expr(
        df,
    ):  # Extract the maximum and minimum values for each num_group and add them as additional features.
        cols = [col for col in df.columns if "num_group" in col]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max, expr_min

    @staticmethod
    def get_exprs(df):  # Execute the above function and return the result.
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
# # File I/O


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.768565Z","iopub.execute_input":"2024-02-10T02:46:14.768903Z","iopub.status.idle":"2024-02-10T02:46:14.781528Z","shell.execute_reply.started":"2024-02-10T02:46:14.768874Z","shell.execute_reply":"2024-02-10T02:46:14.780733Z"}}
def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)

    if depth in [1, 2]:
        maxexprs, minexprs = Aggregator.get_exprs(df)
        df = df.group_by("case_id").agg(*maxexprs, *minexprs)

    return df


def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))

    df = pl.concat(chunks, how="vertical_relaxed")
    if depth in [1, 2]:
        maxexprs, minexprs = Aggregator.get_exprs(df)
        df = df.group_by("case_id").agg(*maxexprs, *minexprs)

    return df


# %% [markdown]
# # Feature Engineering


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.78256Z","iopub.execute_input":"2024-02-10T02:46:14.782843Z","iopub.status.idle":"2024-02-10T02:46:14.793199Z","shell.execute_reply.started":"2024-02-10T02:46:14.782802Z","shell.execute_reply":"2024-02-10T02:46:14.792361Z"}}
def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )

    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    df_base = df_base.pipe(Pipeline.handle_dates)

    return df_base


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.794271Z","iopub.execute_input":"2024-02-10T02:46:14.794592Z","iopub.status.idle":"2024-02-10T02:46:14.80266Z","shell.execute_reply.started":"2024-02-10T02:46:14.794566Z","shell.execute_reply":"2024-02-10T02:46:14.801816Z"}}
def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()

    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)

    df_data[cat_cols] = df_data[cat_cols].astype("category")

    return df_data, cat_cols


# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.803938Z","iopub.execute_input":"2024-02-10T02:46:14.804302Z","iopub.status.idle":"2024-02-10T02:46:14.818011Z","shell.execute_reply.started":"2024-02-10T02:46:14.804269Z","shell.execute_reply":"2024-02-10T02:46:14.817168Z"}}
def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


# %% [markdown]
# # Configuration

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.819111Z","iopub.execute_input":"2024-02-10T02:46:14.819939Z","iopub.status.idle":"2024-02-10T02:46:14.833864Z","shell.execute_reply.started":"2024-02-10T02:46:14.819913Z","shell.execute_reply":"2024-02-10T02:46:14.833129Z"}}
sample = pd.read_csv(
    "/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv"
)
DRY_RUN = (
    True if sample.shape[0] == 10 else False
)  # if num of records of test data is 10, dry-run is enable.
PRESETS = "medium_quality"
MODEL_PATH = "/kaggle/input/home-credit-automl-training/predictor"

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.835388Z","iopub.execute_input":"2024-02-10T02:46:14.835677Z","iopub.status.idle":"2024-02-10T02:46:14.844265Z","shell.execute_reply.started":"2024-02-10T02:46:14.835653Z","shell.execute_reply":"2024-02-10T02:46:14.843523Z"}}
ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"

# %% [markdown]
# # Train Files Read & Feature Engineering

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:14.845257Z","iopub.execute_input":"2024-02-10T02:46:14.845539Z","iopub.status.idle":"2024-02-10T02:46:47.93426Z","shell.execute_reply.started":"2024-02-10T02:46:14.845509Z","shell.execute_reply":"2024-02-10T02:46:47.933428Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:47.939456Z","iopub.execute_input":"2024-02-10T02:46:47.939761Z","iopub.status.idle":"2024-02-10T02:46:58.355676Z","shell.execute_reply.started":"2024-02-10T02:46:47.939735Z","shell.execute_reply":"2024-02-10T02:46:58.354675Z"}}
df_train = feature_eng(**data_store)
print("train data shape:\t", df_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:46:58.356854Z","iopub.execute_input":"2024-02-10T02:46:58.35716Z","iopub.status.idle":"2024-02-10T02:47:01.590123Z","shell.execute_reply.started":"2024-02-10T02:46:58.357131Z","shell.execute_reply":"2024-02-10T02:47:01.589219Z"}}
df_train = df_train.pipe(Pipeline.filter_cols)
print("train data shape:\t", df_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:47:01.59159Z","iopub.execute_input":"2024-02-10T02:47:01.591902Z","iopub.status.idle":"2024-02-10T02:47:21.662418Z","shell.execute_reply.started":"2024-02-10T02:47:01.591874Z","shell.execute_reply":"2024-02-10T02:47:21.661529Z"}}
df_train, cat_cols = to_pandas(df_train)
print(df_train.shape)
df_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:47:21.663768Z","iopub.execute_input":"2024-02-10T02:47:21.66415Z","iopub.status.idle":"2024-02-10T02:47:27.996411Z","shell.execute_reply.started":"2024-02-10T02:47:21.664114Z","shell.execute_reply":"2024-02-10T02:47:27.995506Z"}}
# Load model
predictor = TabularPredictor.load(path=MODEL_PATH)

del data_store
df_train = reduce_mem_usage(df_train)
gc.collect()

# (update_yamashita)add prediction of train_data
X_train = df_train.drop(columns=["case_id", "WEEK_NUM"])
train_data = TabularDataset(df_train)
y_pred_train = predictor.predict_proba(train_data).iloc[:, 1].values

# %% [markdown]
# # Training

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:47:27.997828Z","iopub.execute_input":"2024-02-10T02:47:27.998209Z","iopub.status.idle":"2024-02-10T02:47:28.006362Z","shell.execute_reply.started":"2024-02-10T02:47:27.998171Z","shell.execute_reply":"2024-02-10T02:47:28.00537Z"}}
if DRY_RUN:
    print(f"df_train.shape : {df_train.shape} --> ", end="")
    df_train = df_train.iloc[:500]
    print(df_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:47:28.007618Z","iopub.execute_input":"2024-02-10T02:47:28.007928Z","iopub.status.idle":"2024-02-10T02:47:28.023848Z","shell.execute_reply.started":"2024-02-10T02:47:28.007902Z","shell.execute_reply":"2024-02-10T02:47:28.023034Z"}}
# predictor = TabularPredictor(
#    label="target",
#    problem_type="binary",
#    eval_metric="roc_auc",
#    path="predictor",
# )

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:47:28.024983Z","iopub.execute_input":"2024-02-10T02:47:28.025246Z","iopub.status.idle":"2024-02-10T02:47:28.09791Z","shell.execute_reply.started":"2024-02-10T02:47:28.025222Z","shell.execute_reply":"2024-02-10T02:47:28.097018Z"}}
# weeks = df_train["WEEK_NUM"]
# df_train = df_train.drop(columns=["case_id", "WEEK_NUM"])
# cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
# for idx_train, idx_valid in cv.split(df_train, df_train["target"], groups=weeks):
#    fold_train = df_train.iloc[idx_train]
#    fold_valid = df_train.iloc[idx_valid]
#    train_data = TabularDataset(fold_train)
#    valid_data = TabularDataset(fold_valid)
#    break

# %% [code]
# del df_train # It is deleted to reduce memory usage
# gc.collect()

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-02-10T02:47:28.099022Z","iopub.execute_input":"2024-02-10T02:47:28.099294Z","iopub.status.idle":"2024-02-10T02:48:27.86199Z","shell.execute_reply.started":"2024-02-10T02:47:28.09927Z","shell.execute_reply":"2024-02-10T02:48:27.861267Z"}}
# %%time
# predictor.fit(
#    train_data,
#    tuning_data=valid_data,
#    save_space=True,
#    presets=PRESETS,
#    use_bag_holdout=True,
#    ag_args_fit={'num_gpus': 1},
# )

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:27.863162Z","iopub.execute_input":"2024-02-10T02:48:27.86381Z","iopub.status.idle":"2024-02-10T02:48:28.078208Z","shell.execute_reply.started":"2024-02-10T02:48:27.863781Z","shell.execute_reply":"2024-02-10T02:48:28.07722Z"}}
# gc.collect()

# %% [markdown]
# # Training result

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.079623Z","iopub.execute_input":"2024-02-10T02:48:28.079959Z","iopub.status.idle":"2024-02-10T02:48:28.108593Z","shell.execute_reply.started":"2024-02-10T02:48:28.07993Z","shell.execute_reply":"2024-02-10T02:48:28.107534Z"}}
predictor.leaderboard()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:51:23.265621Z","iopub.execute_input":"2024-02-10T02:51:23.266019Z","iopub.status.idle":"2024-02-10T02:51:23.546485Z","shell.execute_reply.started":"2024-02-10T02:51:23.26598Z","shell.execute_reply":"2024-02-10T02:51:23.545525Z"},"_kg_hide-input":true}
df_lb = predictor.leaderboard()
plt.scatter(df_lb["score_val"], df_lb["model"])
plt.grid()
plt.xlabel("CV(roc_auc)")
plt.ylabel("Model name")
plt.show()

# %% [markdown]
# # Test Files Read & Feature Engineering

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.110424Z","iopub.execute_input":"2024-02-10T02:48:28.110874Z","iopub.status.idle":"2024-02-10T02:48:28.337064Z","shell.execute_reply.started":"2024-02-10T02:48:28.11083Z","shell.execute_reply":"2024-02-10T02:48:28.336186Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.338276Z","iopub.execute_input":"2024-02-10T02:48:28.338615Z","iopub.status.idle":"2024-02-10T02:48:28.377767Z","shell.execute_reply.started":"2024-02-10T02:48:28.338586Z","shell.execute_reply":"2024-02-10T02:48:28.376924Z"}}
df_test = feature_eng(**data_store)
print("test data shape:\t", df_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.37902Z","iopub.execute_input":"2024-02-10T02:48:28.379298Z","iopub.status.idle":"2024-02-10T02:48:28.386911Z","shell.execute_reply.started":"2024-02-10T02:48:28.379272Z","shell.execute_reply":"2024-02-10T02:48:28.386002Z"}}
df_test = df_test.select([col for col in df_test.columns if col != "target"])
print("test data shape:\t", df_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.388262Z","iopub.execute_input":"2024-02-10T02:48:28.388811Z","iopub.status.idle":"2024-02-10T02:48:28.455392Z","shell.execute_reply.started":"2024-02-10T02:48:28.388783Z","shell.execute_reply":"2024-02-10T02:48:28.454505Z"}}
df_test, cat_cols = to_pandas(df_test, cat_cols)  # cat_cols was created by train data

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.456707Z","iopub.execute_input":"2024-02-10T02:48:28.457875Z","iopub.status.idle":"2024-02-10T02:48:28.482038Z","shell.execute_reply.started":"2024-02-10T02:48:28.457846Z","shell.execute_reply":"2024-02-10T02:48:28.480922Z"}}
print(df_test.shape)
df_test.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.483324Z","iopub.execute_input":"2024-02-10T02:48:28.483647Z","iopub.status.idle":"2024-02-10T02:48:28.943783Z","shell.execute_reply.started":"2024-02-10T02:48:28.48361Z","shell.execute_reply":"2024-02-10T02:48:28.942867Z"}}
del data_store
df_test = reduce_mem_usage(df_test)
gc.collect()

# %% [markdown]
# # Prediction

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:28.94502Z","iopub.execute_input":"2024-02-10T02:48:28.945314Z","iopub.status.idle":"2024-02-10T02:48:29.071279Z","shell.execute_reply.started":"2024-02-10T02:48:28.945288Z","shell.execute_reply":"2024-02-10T02:48:29.070222Z"}}
X_test = df_test.drop(columns=["case_id", "WEEK_NUM"])
test_data = TabularDataset(df_test)
y_pred = predictor.predict_proba(test_data).iloc[:, 1].values

# %% [markdown]
# # Submission

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:29.072777Z","iopub.execute_input":"2024-02-10T02:48:29.073122Z","iopub.status.idle":"2024-02-10T02:48:29.080065Z","shell.execute_reply.started":"2024-02-10T02:48:29.073093Z","shell.execute_reply":"2024-02-10T02:48:29.079266Z"}}
df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:29.083243Z","iopub.execute_input":"2024-02-10T02:48:29.083682Z","iopub.status.idle":"2024-02-10T02:48:29.093817Z","shell.execute_reply.started":"2024-02-10T02:48:29.083654Z","shell.execute_reply":"2024-02-10T02:48:29.092877Z"}}
print("Check null: ", df_subm["score"].isnull().any())

df_subm.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-02-10T02:48:29.095227Z","iopub.execute_input":"2024-02-10T02:48:29.095588Z","iopub.status.idle":"2024-02-10T02:48:29.105693Z","shell.execute_reply.started":"2024-02-10T02:48:29.095559Z","shell.execute_reply":"2024-02-10T02:48:29.104736Z"}}
df_subm.to_csv("submission.csv")