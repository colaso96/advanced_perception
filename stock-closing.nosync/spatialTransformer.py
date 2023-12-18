import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

os.chdir("../../..")
import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from sklearn.preprocessing import StandardScaler
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#Initialize Data
target_series = pd.read_csv('/content/drive/MyDrive/train_target_series.csv')
valid_series = pd.read_csv('/content/drive/MyDrive/valid_target_series.csv')

#Reduce memory usage by converting to reduced data types
def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)

    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f"Decreased by {decrease:.2f}%")

    return df

#Initialize stardscalar on the target series
scaler =  StandardScaler()
scaler.fit(target_series)

#remove null values, reduce memory & transformer on the target series 
target_series.fillna(0, inplace=True)
target_series = reduce_mem_usage(target_series, verbose=1)
target_series = scaler.fit_transform(target_series)

#remove null values, reduce memory & transformer on the valid series 
valid_series.fillna(0, inplace=True)
valid_series = reduce_mem_usage(valid_series, verbose=1)
valid_series = scaler.transform(valid_series)

target_series = pd.DataFrame(target_series)
valid_series = pd.DataFrame(valid_series)

#Converting the column types from ints to string to be passed into the model & add a time stamp column & dummy 
#to pass into the group field for multitarget prediction in the TFT
target_series.columns = target_series.columns.astype(str)
target_series.reset_index(level=0, inplace=True)
target_series.rename(columns={'index': 'time_step'}, inplace=True)
target_series["dummy_constant"] = 1
#target_series.head()

#Converting the column types from ints to string to be passed into the model & add a time stamp column & dummy 
#to pass into the group field for multitarget prediction in the TFT
valid_series.columns = valid_series.columns.astype(str) 
valid_series.reset_index(level=0, inplace=True)
valid_series.rename(columns={'index': 'time_step'}, inplace=True)
valid_series["dummy_constant"] = 1
#valid_series.head()

# Define the target columns as all columns except the timestamp and dummy constant
target_columns = [col for col in target_series.columns if col not in ["time_step", "dummy_constant"]]
print(target_columns)


# Create a TimeSeriesDataSet using all stock windows as the target column, the dummy constant passed into group_ids 
training = TimeSeriesDataSet(
    target_series,
    time_idx="time_step",
    target=target_columns,
    group_ids=["dummy_constant"],
    min_encoder_length=24,
    max_encoder_length=48,
    min_prediction_length=1,
    max_prediction_length=200, #setting prediciton length of all 200 stocks
    #time_varying_unknown_reals=target_columns,
    #time_varying_known_reals=["seconds_in_bucket"],
    allow_missing_timesteps=True, #need to run
    add_relative_time_idx = True, #needed 
)

validation = TimeSeriesDataSet.from_dataset(training, valid_series, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="gpu",
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)

#Setup mock TFT to find the optimal learning rate
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    optimizer="Ranger",
    reduce_on_plateau_patience=10,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate
from lightning.pytorch.tuner import Tuner

res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()


## configure network and trainer
LEARNING_RATE = res.suggestion()

# configure network and trainer allowing for logging and learning rate monitors
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

#Setup the trainer with max_epochs optimization  & callbacks
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    #fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

#Setup the TFT from the TimeSeriesDataset using the optimal learning rate derived
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    #log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
print(len(predictions.y))

#importing the unscaled df to fit the scalar
train_unscaled = pd.read_csv('/content/drive/MyDrive/train_target_series.csv')

#Inverse transform output to calculate MAE
scaler =  StandardScaler()
scaler.fit(train_unscaled)
trans_y = scaler.inverse_transform(predictions.y[0][0].cpu())
trans_out = scaler.inverse_transform(predictions.output[0][0].cpu().unsqueeze(0))

print(type(trans_y), len(trans_out[0]))
mae = mean_absolute_error(trans_y, trans_out)

print("Mean Absolute Error:", mae)
