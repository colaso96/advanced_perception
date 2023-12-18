# Bronte Sihan Li, Cole Crescas Dec 2023
# CS7180
import torch
from pathlib import Path
import os
from tqdm import tqdm
from dataset import StockDataset
from models import (
    ResCNN,
    LSTMRegressor,
    SimpleTransformer,
    TimeSeriesTransformer,
    ThreeLayerTransformer,
)

model_mapping = {
    'rescnn': ResCNN(target_series=False),
    'rescnn_ts': ResCNN(target_series=True),
    'lstm': LSTMRegressor(),
    'lstm_ts': LSTMRegressor(input_size=200, output_size=200),
    'transformer': SimpleTransformer(),
    'transformer_ts': SimpleTransformer(feature_num=200, dropout_rate=0.25),
    'transformer_improved': TimeSeriesTransformer(feature_num=200, dropout_rate=0.25),
    'ThreeLayerTransformer': ThreeLayerTransformer(feature_num=200, dropout_rate=0.25),
}

TEST_DATA_DIR = Path('./data/train.csv')


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    batch_size,
    criterion,
    n_val,
    scaler,
    dataset_type='ts',
    amp=False,
    save_predictions=False,
):
    """
    Evaluate a model on a validation set.
    """
    save_dir = Path(f'predictions_{net.__class__.__name__}')
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    net.to(device=device)
    torch.set_grad_enabled(False)
    num_val_batches = n_val // batch_size
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in tqdm(
            enumerate(dataloader),
            total=num_val_batches,
            desc='Validation round',
            unit='batch',
            leave=False,
        ):
            features, target = batch
            features = features.to(device=device)
            target = target.to(device=device)
            pred = net(features)
            if dataset_type == 'ts':
                pred = scaler.inverse_transform(pred.cpu())
                target = scaler.inverse_transform(target.cpu())
                pred = torch.from_numpy(pred).to(device)
                target = torch.from_numpy(target).to(device)
            # compute the loss
            val_loss += criterion(
                pred,
                target,
            ).item()

    net.train()
    return val_loss / max(num_val_batches, 1)


def calculate_inference_time(model):
    """
    Calculates inference time / throughput for a given model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(32, 1, 200, 200).to(device)
    model.to(device)
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * 32) / total_time
    print('Final Throughput:', Throughput)


def main():
    for model_name in [
        'lstm_ts',
        'transformer_ts',
        'transformer_improved',
        'ThreeLayerTransformer',
    ]:
        model = model_mapping[model_name]
        model.eval()
        print(model_name)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {params}')
        calculate_inference_time(model)


if __name__ == '__main__':
    main()
