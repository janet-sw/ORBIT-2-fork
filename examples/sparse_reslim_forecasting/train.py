"""Train the minimal Sparse-Reslim deterministic forecasting example."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from .config import ConfigError, load_forecast_config
    from .model import SparseReslim
    from .utils import seed_everything
except ImportError:  # Support `python examples/.../train.py` from the repo root.
    from config import ConfigError, load_forecast_config
    from model import SparseReslim
    from utils import seed_everything


def _as_time_lat_lon(array: np.ndarray, variable: str) -> np.ndarray:
    """Normalize common ORBIT NPZ layouts to [time, latitude, longitude]."""
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            f"{variable!r} must have shape [T,H,W] or [T,1,H,W]; got {array.shape}"
        )
    return array


def _load_normalization(root: Path, variables: tuple[str, ...]):
    means_path = root / "normalize_mean.npz"
    stds_path = root / "normalize_std.npz"
    if not means_path.exists() or not stds_path.exists():
        raise FileNotFoundError(
            "Expected normalize_mean.npz and normalize_std.npz in the ERA5 root"
        )
    with np.load(means_path) as mean_file, np.load(stds_path) as std_file:
        means = {name: float(np.asarray(mean_file[name]).reshape(-1)[0]) for name in variables}
        stds = {name: float(np.asarray(std_file[name]).reshape(-1)[0]) for name in variables}
    if any(value == 0 for value in stds.values()):
        raise ValueError("normalization standard deviations must be non-zero")
    return means, stds


class ERA5ForecastDataset(IterableDataset):
    """Stream direct-forecast pairs from ORBIT-style yearly NPZ files."""

    def __init__(
        self,
        root: Path,
        split: str,
        input_variables: tuple[str, ...],
        output_variables: tuple[str, ...],
        *,
        history: int,
        window: int,
        pred_range: int,
        shuffle_files: bool,
    ) -> None:
        super().__init__()
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found in {root / split}")
        self.input_variables = input_variables
        self.output_variables = output_variables
        all_variables = tuple(dict.fromkeys(input_variables + output_variables))
        self.means, self.stds = _load_normalization(root, all_variables)
        self.history = history
        self.window = window
        self.pred_range = pred_range
        self.shuffle_files = shuffle_files

    def __iter__(self):
        files = list(self.files)
        if self.shuffle_files:
            random.shuffle(files)
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        for path in files:
            with np.load(path) as data:
                required = set(self.input_variables + self.output_variables)
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"{path} is missing variables: {sorted(missing)}")
                arrays = {
                    name: _as_time_lat_lon(data[name], name)
                    for name in required
                }

            total_steps = min(array.shape[0] for array in arrays.values())
            first_input = (self.history - 1) * self.window
            last_input = total_steps - self.pred_range
            for time_index in range(first_input, last_input):
                history_indices = [
                    time_index - offset * self.window
                    for offset in reversed(range(self.history))
                ]
                inputs = np.stack(
                    [
                        np.stack(
                            [
                                (arrays[name][index] - self.means[name])
                                / self.stds[name]
                                for name in self.input_variables
                            ],
                            axis=0,
                        )
                        for index in history_indices
                    ],
                    axis=0,
                )
                target_index = time_index + self.pred_range
                targets = np.stack(
                    [
                        (arrays[name][target_index] - self.means[name])
                        / self.stds[name]
                        for name in self.output_variables
                    ],
                    axis=0,
                )
                yield torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(
                    targets.astype(np.float32)
                )


def _infer_image_size(root: Path, variable: str) -> tuple[int, int]:
    candidates = sorted((root / "train").glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No training NPZ files found in {root / 'train'}")
    with np.load(candidates[0]) as data:
        sample = _as_time_lat_lon(data[variable], variable)
    return int(sample.shape[-2]), int(sample.shape[-1])


def _build_model(args, img_size: tuple[int, int]) -> SparseReslim:
    return SparseReslim(
        args.input_vars,
        args.output_vars,
        img_size,
        history=args.history,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        keep_ratio=args.keep_ratio,
        num_dense_early=args.num_dense_early,
        num_sparse_middle=args.num_sparse_middle,
        dropout=args.dropout,
    )


def _select_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "trainer.accelerator is 'gpu', but PyTorch cannot access a GPU"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device, data_type: str):
    if device.type == "cuda" and data_type == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _run_epoch(
    model: SparseReslim,
    loader,
    device: torch.device,
    data_type: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, data_type):
                forecast = model(inputs)
                loss = F.mse_loss(forecast, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite MSE loss: {loss.item()}")
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The data loader did not produce any forecast samples")
    return total_loss / total_samples


def _load_checkpoint(path: Path, model, optimizer=None):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _save_checkpoint(
    path: Path,
    model: SparseReslim,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mse": best_val_mse,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def run_smoke_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseReslim(
        variables=("2m_temperature",),
        output_variables=("2m_temperature",),
        img_size=(8, 16),
        patch_size=2,
        embed_dim=32,
        depth=4,
        num_heads=4,
        keep_ratio=0.25,
        num_dense_early=1,
        num_sparse_middle=2,
    ).to(device)
    inputs = torch.randn(2, 1, 1, 8, 16, device=device)
    targets = torch.randn(2, 1, 8, 16, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_mse = _run_epoch(
        model,
        [(inputs, targets)],
        device,
        "float32",
        optimizer=optimizer,
    )
    forecast = model(inputs)
    assert forecast.shape == targets.shape
    assert model.last_sparse_token_count == 8  # 25% of 32 patch tokens.
    print(
        "Smoke test passed:",
        f"device={device},",
        f"forecast={tuple(forecast.shape)},",
        f"sparse_tokens={model.last_sparse_token_count}/{model.num_patches},",
        f"train_mse={train_mse:.4f}",
    )


def run_training(args) -> None:
    if args.devices != 1:
        raise SystemExit(
            "This native PyTorch example currently supports one device. "
            "DDP and FSDP will be added as separate parallelism work."
        )
    device = _select_device(args.accelerator)
    seed_everything(args.seed)
    root = args.era5_dir.expanduser().resolve()
    img_size = _infer_image_size(root, args.input_vars[0])
    if img_size[0] % args.patch_size or img_size[1] % args.patch_size:
        raise SystemExit(
            f"Image size {img_size} is not divisible by patch size {args.patch_size}"
        )

    common_dataset_args = dict(
        root=root,
        input_variables=args.input_vars,
        output_variables=args.output_vars,
        history=args.history,
        window=args.window,
        pred_range=args.pred_range,
    )
    train_dataset = ERA5ForecastDataset(
        split="train", shuffle_files=True, **common_dataset_args
    )
    val_dataset = ERA5ForecastDataset(
        split="val", shuffle_files=False, **common_dataset_args
    )
    test_dataset = ERA5ForecastDataset(
        split="test", shuffle_files=False, **common_dataset_args
    )

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    network = _build_model(args, img_size).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = args.output_dir.expanduser().resolve()
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    start_epoch = 0
    best_val_mse = float("inf")
    epochs_without_improvement = 0

    if args.pretrain:
        _load_checkpoint(Path(args.pretrain).expanduser(), network)
        print(f"Loaded pretrained weights from {args.pretrain}", flush=True)
    if args.checkpoint:
        checkpoint = _load_checkpoint(
            Path(args.checkpoint).expanduser(), network, optimizer
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_mse = float(checkpoint.get("best_val_mse", best_val_mse))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)

    print(f"Training on {device} with native PyTorch", flush=True)
    for epoch in range(start_epoch, args.max_epochs):
        train_mse = _run_epoch(
            network,
            loader(train_dataset),
            device,
            args.data_type,
            optimizer=optimizer,
            max_batches=args.limit_train_batches,
        )
        val_mse = _run_epoch(
            network,
            loader(val_dataset),
            device,
            args.data_type,
            max_batches=args.limit_val_batches,
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.max_epochs:03d} "
            f"train/mse={train_mse:.6f} val/mse={val_mse:.6f}",
            flush=True,
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint,
                network,
                optimizer,
                epoch,
                best_val_mse,
                epochs_without_improvement,
            )
            print(f"Saved best checkpoint to {best_checkpoint}", flush=True)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {epochs_without_improvement} "
                    "epochs without validation improvement",
                    flush=True,
                )
                break

    if not best_checkpoint.exists():
        raise RuntimeError(
            "Training completed without producing a best checkpoint. "
            "Check max_epochs and checkpoint settings."
        )
    _load_checkpoint(best_checkpoint, network)
    test_mse = _run_epoch(
        network,
        loader(test_dataset),
        device,
        args.data_type,
        max_batches=args.limit_test_batches,
    )
    print(f"test/mse={test_mse:.6f}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal Sparse-Reslim deterministic ERA5 forecasting example"
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="STORM-style YAML configuration file",
    )
    parser.add_argument("--smoke-test", action="store_true")
    cli_args = parser.parse_args(argv)
    if cli_args.smoke_test:
        return cli_args
    if cli_args.config is None:
        parser.error("CONFIG is required unless --smoke-test is used")
    try:
        return load_forecast_config(cli_args.config)
    except ConfigError as error:
        parser.error(str(error))


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        run_smoke_test()
    else:
        run_training(arguments)
