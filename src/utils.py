import logging
import warnings
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MetricTracker(pl.Callback):
    def __init__(self):
        self.collection = []

    def on_validation_epoch_end(self, trainer, module):
        elogs = deepcopy(trainer.logged_metrics)  # access it here
        self.collection.append(elogs)


class MetricsMixin(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.training_step_outputs.append((loss, x_batch, y_batch))
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.validation_step_outputs.append((loss, x_batch, y_batch))
        return loss

    def on_train_epoch_end(self):
        step_outputs = self.training_step_outputs
        self._epoch_end(step_outputs, mode="train")

    def on_validation_epoch_end(self):
        step_outputs = self.validation_step_outputs
        self._epoch_end(step_outputs, mode="val")

    def _epoch_end(self, step_outputs, mode):
        res_dict = {}
        # self.eval()
        test_loss = 0
        num_batches = 0

        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for loss, x_batch, y_batch in step_outputs:
                y_pred = self(x_batch)
                all_y.append(y_batch)

                test_loss += loss.item()

                # convert logits to labels
                y_pred = y_pred.reshape(-1)
                y_pred = (y_pred > 0).int()
                all_y_pred.append(y_pred)
                num_batches += 1

        test_loss /= num_batches
        res_dict["loss"] = test_loss
        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        res_dict = res_dict | get_metrics(y_true=all_y.cpu().numpy(), y_pred=all_y_pred.cpu().numpy())

        for k, v in res_dict.items():
            self.log(f"{mode}_{k}", np.float32(v))

        if mode == "val":
            self.validation_step_outputs.clear()
        else:
            self.training_step_outputs.clear()


# track grad norm in ligthning 2.0
# https://github.com/Lightning-AI/lightning/issues/1462#issuecomment-1190253742
class GradNormCallback(pl.Callback):
    """
    Logs the gradient norm.
    """

    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("my_model/grad_norm", GradNormCallback.gradient_norm(model))


_dataset_name_to_filename = {
    "KPI": "KPI-Anomaly-Detection/Preliminary_dataset/train.parquet",
    "FI2010": "FI2010/all_with_anomalies.parquet",
    # "NAB": "NAB/processed_ambient_temperature_system_failure.parquet",
    "NAB": "NAB/processed_nyc_taxi.parquet",
}


def convert_to_windows(x, window_size: int):
    # convert input time series x to
    # time series of lags, first window_size observations are dropped
    windows = []
    for i in range(window_size, len(x)):
        w = x[i - window_size : i]
        windows.append(w)
    windows = np.array(windows)
    return windows


def prepare_dataset_for_evaluation(
    dataset_name: str,
    window_size: int,
    train_proportion: float,
    batch_size: int = 2 * 4096,
    root_data_dir=Path("../data"),
    seed: int = 1,
):
    if seed is not None:
        seed_everything(1)
    dataset_filename = _dataset_name_to_filename[dataset_name]
    df = pd.read_parquet(root_data_dir / dataset_filename)
    if dataset_name == "KPI":
        df = df[df["KPI ID"] == "02e99bd4f6cfb33f"]
        df = df[df.index > 1.49 * 1e9]
        df = df.query("timestamp < 1496538120").copy()
        df["time"] = df.index - df.index[0]
        df.dropna(inplace=True)
    elif dataset_name == "FI2010":
        df = df.query("stock==1 & day==1 & train==1").copy()
    elif dataset_name == "NAB":
        pass
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    df["value_diff"] = df["value"].diff()
    tr_cols = ["value", "value_diff"]
    df = df.dropna()
    df.dropna(inplace=True)

    logger.info(f"Dataset shape: {df.shape}")

    x = df[tr_cols].values.copy()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    # x_scaled = x
    x = convert_to_windows(x, window_size)
    y = df["target"].values
    y = y[window_size:]

    x = torch.Tensor(x).float()
    y = torch.Tensor(y).float()

    logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")

    train_idx = int(len(x) * train_proportion)

    logger.info(f"train_idx: {train_idx}")

    tr, va = x[:train_idx], x[train_idx:]
    ytr, yva = y[:train_idx], y[train_idx:]
    tr_dataset = TensorDataset(tr, ytr)
    va_dataset = TensorDataset(va, yva)

    logger.warning(f"y_tr mean {ytr.mean()}, y_va mean {yva.mean()}")

    logger.info(f"tr_dataset: {len(tr_dataset)}, va_dataset: {len(va_dataset)}")

    if seed is not None:
        seed_everything(seed)

    tr_dl = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
    va_dl = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    return tr_dl, va_dl, tr_cols


# a function to get accuracy/precision/recall/AUC/F1 from y_true and y_pred
def get_metrics(y_true, y_pred):
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        accuracy = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    res["accuracy"] = accuracy
    res["precision"] = precision
    res["recall"] = recall
    res["auc"] = auc
    res["f1"] = f1
    return res


def get_metrics_from_tracker(metrics_tracker: MetricTracker):
    res = deepcopy(metrics_tracker.collection)
    for r in res:
        for k, v in r.items():
            if type(v) == torch.Tensor:
                r[k] = np.float32(v.cpu())
            else:
                r[k] = np.float32(v)
    res = pd.DataFrame(res)
    return res


def plot_metrics_from_tracker(res, dataset_name=None, filename=None):
    import scienceplots  # noqa # pylint: disable=unused-import

    # with plt.style.context(["science", "ieee"]):
    # with plt.style.context(["ieee"]):
    with plt.style.context(["default"]):
        plt.rcParams.update({"axes.grid": True})

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        if dataset_name is not None or dataset_name != "":
            fig.suptitle(dataset_name)

        plt.sca(axs[0, 0])
        plt.plot(res["train_loss"], label="Train")
        plt.plot(res["val_loss"], label="Validation")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.sca(axs[0, 1])
        plt.plot(res["train_f1"], label="Train")
        plt.plot(res["val_f1"], label="Validation")
        plt.title("F1")
        plt.xlabel("Epoch")
        plt.sca(axs[1, 0])
        plt.plot(res["train_precision"], label="Train")
        plt.plot(res["val_precision"], label="Validation")
        plt.title("Precision")
        plt.xlabel("Epoch")
        plt.sca(axs[1, 1])
        plt.plot(res["train_recall"], label="Train")
        plt.plot(res["val_recall"], label="Validation")
        plt.title("Recall")
        plt.xlabel("Epoch")
        fig.tight_layout()

        if filename is not None:
            plt.savefig(filename)
