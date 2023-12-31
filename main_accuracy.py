import argparse
import logging

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateFinder
import torch

import torch.nn as nn

from src.utils import MetricTracker
from src.transformer import TransformerAnomalyDetector
from src.transformer import LinearRegressionAnomalyDetector
from src.transformer import LinearTransformerAnomalyDetector
from src.utils import GradNormCallback
from src.utils import (
    get_metrics_from_tracker,
    plot_metrics_from_tracker,
    prepare_dataset_for_evaluation,
)


from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)-12s %(asctime)s: %(levelname)-8s %(message)s",
    handlers=[logging.FileHandler("main_accuracy.log", mode="w"), logging.StreamHandler()],
)
logger.setLevel(logging.INFO)

DEFAULT_ROOT_DATA_DIR = Path("data/")
batch_size = 2 * 4096
window_size = 8
train_proportion = 0.9

override_dicts = {
    ("Transformer", "NAB"): {
        "epochs": 300,
        "gradient_clip_val": 0.01,
        "max_lr": 0.1,
    },
    ("Transformer", "FI2010"): {
        "epochs": 101,
        "gradient_clip_val": 0.01,
        "max_lr": 0.01,
    },
    ("LinearTransformer", "FI2010"): {
        "epochs": 271,
        # "gradient_clip_val": 0.01,
        # "max_lr": 0.01,
    },
}


def get_model(model_name, input_dim, window_size=8, override_params_dict=None):
    ret_dict = {"model": None, "epochs": None, "gradient_clip_val": None, "max_lr": 1}
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))
    if model_name == "Transformer":
        block_args = {
            "input_dim": 8,
            "num_heads": 2,
            "dim_feedforward": 2 * 8,
            "num_layers": 1,
            "enable_layer_norm": False,
        }
        positional_encoder_args = {
            "enable": False,
            "max_len": window_size,
        }

        model_params = {
            "input_dim": input_dim,
            "block_input_dim": block_args["input_dim"],
            "block_args": block_args,
            "num_layers": block_args["num_layers"],
            "positional_encoder_args": positional_encoder_args,
            "learning_rate": 1e-1,
            "dropout": 0.0,
            "loss_fn": loss_fn,
        }

        model = TransformerAnomalyDetector(
            **model_params,
        )
        ret_dict["model"] = model
        ret_dict["epochs"] = 100
        ret_dict["gradient_clip_val"] = 0.1
    elif model_name == "LinearRegression":
        input_dim = input_dim * window_size

        model_params = {
            "input_dim": input_dim,
            "learning_rate": 1e-5,
            "loss_fn": loss_fn,
        }

        model = LinearRegressionAnomalyDetector(
            **model_params,
        )
        ret_dict["model"] = model
        ret_dict["epochs"] = 260
        ret_dict["gradient_clip_val"] = None
    elif model_name == "LinearTransformer":
        block_args = {
            "input_dim": 8,
            "dim_feedforward": 2 * 8,
            "num_layers": 1,
            "enable_layer_norm": False,
        }
        positional_encoder_args = {
            "enable": False,
            "max_len": window_size,
        }

        model_params = {
            "input_dim": input_dim,
            "block_input_dim": block_args["input_dim"],
            "block_args": block_args,
            "num_layers": block_args["num_layers"],
            "positional_encoder_args": positional_encoder_args,
            "learning_rate": 1e-1,
            "dropout": 0.0,
            "loss_fn": loss_fn,
        }

        model = LinearTransformerAnomalyDetector(
            **model_params,
        )
        ret_dict["model"] = model
        ret_dict["epochs"] = 100
        ret_dict["gradient_clip_val"] = 1.2
        ret_dict["max_lr"] = 0.1
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if override_params_dict is not None:
        logger.info(f"Overriding model params with: {override_params_dict}")
        for k, v in override_params_dict.items():
            ret_dict[k] = v

    return ret_dict


def main(model_name, dataset_name, plots_root_dir=Path("tex/Figures/plots"), root_data_dir=DEFAULT_ROOT_DATA_DIR):
    seed_everything(1)

    tr_dl, va_dl, tr_cols = prepare_dataset_for_evaluation(
        dataset_name=dataset_name,
        window_size=window_size,
        train_proportion=train_proportion,
        batch_size=batch_size,
        root_data_dir=root_data_dir,
    )

    override_params_dict = override_dicts.get((model_name, dataset_name), None)

    model_dict = get_model(model_name, input_dim=len(tr_cols), override_params_dict=override_params_dict)
    model = model_dict["model"]
    epochs = model_dict["epochs"]
    gradient_clip_val = model_dict["gradient_clip_val"]
    max_lr = model_dict["max_lr"]
    logger.info(
        f"Training model: {model_name} on dataset: {dataset_name} for {epochs} epochs. Input dim: {len(tr_cols)}"
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        log_every_n_steps=1,
        callbacks=[LearningRateFinder(max_lr=max_lr), MetricTracker(), GradNormCallback()],
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model, tr_dl, va_dl)

    # for plots
    mt = trainer.callbacks[1]
    res = get_metrics_from_tracker(mt)
    if plots_root_dir is not None:
        logger.info(f"Plotting metrics to {plots_root_dir}")
        plot_metrics_from_tracker(
            res, dataset_name=dataset_name, filename=plots_root_dir / f"{model_name}_{dataset_name}.png"
        )

    return_dict = {"model": model, "metrics": res}
    return return_dict


if __name__ == "__main__":
    logger.info("Starting main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Transformer", help="model name")
    parser.add_argument("--dataset", type=str, default="KPI", help="dataset name")

    args = parser.parse_args()

    metrics_res = main(model_name=args.model, dataset_name=args.dataset)

    logger.info("Finished main")
