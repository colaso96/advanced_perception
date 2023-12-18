# Bronte Sihan Li, Cole Crescas Dec 2023
# CS7180
import wandb
import gc
import torch
import os
import logging
import random
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from dataset import (
    StockDataset,
    DATA_FILE_DIR,
    TargetTimeSeriesDataset,
    TRAIN_TARGET_SERIES_DATA_FILE_DIR,
    VALIDATION_TARGET_SERIES_DATA_FILE_DIR,
)
from evaluate import evaluate, model_mapping
import pytorch_warmup as warmup
from wandb import Artifact
from tqdm import tqdm
from torch.utils.data import random_split
from typing import Literal
from sklearn.preprocessing import StandardScaler
import pandas as pd

dir_checkpoint = Path('checkpoints')
log_dir = Path('logs')
seed = 42


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(
            ' | '.join(
                [
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors",
                ]
                + [f"{k} {v}" for k, v in group_hps.items()]
            )
        )

    return optimizer, scheduler


def train(
    model,
    device,
    dataset_type: Literal['full', 'ts'] = 'ts',
    window_size: int = 55,
    dir_checkpoint: Path = dir_checkpoint,
    starting_epoch: int = 1,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    amp: bool = False,
    optimizer: str = 'adamw',
    optimizer_state_dict=None,
    weight_decay: float = 0.05,
    momentum: float = 0.9,
    gradient_clipping: float = 1.0,
    loss_function: str = 'mae',
    activation: str = 'relu',
    use_warmup: bool = True,
    warmup_lr_init: float = 5e-7,
):
    # Create dataset and dataloader
    if dataset_type == 'full':
        dataset = StockDataset(DATA_FILE_DIR, window_size=window_size)
        total_length = len(dataset)
        n_val = int(val_percent * total_length)
        n_train = total_length - n_val

        # Use the calculated lengths to split the dataset
        train_set, val_set = random_split(dataset, [n_train, n_val])
    elif dataset_type == 'ts':
        scaler = StandardScaler()
        scaler.fit(pd.read_csv(TRAIN_TARGET_SERIES_DATA_FILE_DIR).to_numpy())
        train_set = TargetTimeSeriesDataset(
            TRAIN_TARGET_SERIES_DATA_FILE_DIR, window_size=window_size, scaler=scaler
        )
        val_set = TargetTimeSeriesDataset(
            VALIDATION_TARGET_SERIES_DATA_FILE_DIR,
            window_size=window_size,
            scaler=scaler,
        )
        total_length = len(train_set) + len(val_set)

    # Print lengths for verification
    print("Total dataset length:", total_length)
    print("Train set length:", len(train_set))
    print("Validation set length:", len(val_set))
    n_train = len(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    # Initialize logging
    experiment = wandb.init(
        entity='advanced-perception',
        project='optiver',
        resume='allow',
        anonymous='must',
    )
    experiment.config.update(
        dict(
            starting_epoch=starting_epoch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            amp=amp,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            loss_function=loss_function,
            activation=activation,
            use_warmup=use_warmup,
        )
    )
    os.makedirs(log_dir, exist_ok=True)
    logging.info(
        f"""
        Starting epoch: {starting_epoch}
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {learning_rate}
        Validation percent: {val_percent}
        Save checkpoint: {save_checkpoint}
        Mixed precision training: {amp}
        Optimizer: {optimizer}
        Weight decay: {weight_decay}
        Momentum: {momentum}
        Loss function: {loss_function}
        Activation: {activation}
        Use warmup: {use_warmup}
        """
    )
    num_steps = len(train_loader) * epochs
    # Set up optimizer
    if optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
            foreach=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps
        )
    elif optimizer == 's4':
        optimizer, lr_scheduler = setup_optimizer(
            model, lr=learning_rate, weight_decay=weight_decay, epochs=epochs
        )
    else:
        raise ValueError(f'Optimizer {optimizer} not supported.')
    # Set up loss function
    if loss_function == 'mae':
        criterion = torch.nn.L1Loss()
    elif loss_function == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f'Loss function {loss_function} not supported.')
    # Set up scheduler
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
    global_step = 0
    if use_warmup:
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    else:
        warmup_scheduler = None

    # Train model
    for epoch in range(1, epochs + 1):
        torch.set_grad_enabled(True)
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='sample') as pbar:
            for i, data in enumerate(train_loader):
                features, target = data
                features = features.to(device)
                target = target.to(device)

                pred = model(features)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if warmup_scheduler:
                    with warmup_scheduler.dampening():
                        lr_scheduler.step()
                else:
                    lr_scheduler.step()
                pbar.update(features.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {'train loss': loss.item(), 'step': global_step, 'epoch': epoch}
                )
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_loss = evaluate(
                            model,
                            val_loader,
                            device,
                            batch_size,
                            criterion,
                            scaler=scaler,
                            dataset_type=dataset_type,
                            n_val=len(val_set),
                        )
                        logging.info('Validation Loss: %s', val_loss)
                        experiment.log(
                            {'val loss': val_loss, 'step': global_step, 'epoch': epoch}
                        )
                        if save_checkpoint:
                            save_dir = Path(
                                dir_checkpoint, model.__class__.__name__, dataset_type
                            )
                            save_dir.mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            torch.save(
                                {
                                    'model_state_dict': state_dict,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch,
                                },
                                f'{str(save_dir)}/checkpoint_epoch{epoch+starting_epoch}_{val_loss}.pth',
                            )
                            experiment.log_artifact(
                                Artifact(
                                    f'checkpoint_epoch{epoch+starting_epoch}_{val_loss}.pth',
                                    type='model',
                                    metadata=dict(
                                        model_type=model.__class__.__name__,
                                        starting_epoch=epoch + starting_epoch,
                                        val_loss=val_loss,
                                    ),
                                )
                            )
                            logging.info(f'Checkpoint {epoch+starting_epoch} saved!')
                            gc.collect()


def main(
    epochs=10,
    batch_size=128,
    lr=0.001,
    val_percent=0.2,
    amp=False,
    model_type: Literal['s4', 'rescnn', 'lstm', 'transformer'] = 's4',
    save_checkpoint=False,
):
    optimizer = 's4' if model_type == 's4' else 'adamw'
    optimizer_state_dict = None
    loss_function = 'mae'
    activation = 'relu'

    seed_all(seed=seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = model_mapping[model_type]
    model.to(device=device)
    logging.info(f'Model: {model.__class__.__name__}')

    train(
        model,
        device,
        window_size=100,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        val_percent=val_percent,
        amp=amp,
        optimizer=optimizer,
        optimizer_state_dict=optimizer_state_dict,
        loss_function=loss_function,
        activation=activation,
        save_checkpoint=save_checkpoint,
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_percent', type=float, default=0.2)
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--model_type', type=str, default='lstm_ts')
    parser.add_argument('--save_checkpoint', action='store_false')
    main(**vars(parser.parse_args()))
