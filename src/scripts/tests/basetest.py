import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.dataset_openfwi import OpenFWI


def _get_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    return float(value)


def base_test(model, conf, fast_run=False):
    test_dataset_list = []
    for dataset in conf.datasets.dataset_name:
        dataset = OpenFWI(root_dir="data/openfwi", use_data=conf.datasets.use_data, datasets=(dataset,),
                          use_normalize=conf.datasets.use_normalize)
        total_size = len(dataset)
        test_size = int(0.1 * total_size)  # Use the final 10% as a deterministic test split.
        test_idx = list(range(total_size - test_size, total_size))
        # test_idx = list(range(test_size))

        test_dataset_list.append(Subset(dataset, test_idx))

    test_set = ConcatDataset(test_dataset_list)

    test_loader = DataLoader(test_set, batch_size=conf.training.dataloader.batch_size,
                             num_workers=conf.training.dataloader.num_workers, persistent_workers=True, pin_memory=True,
                             prefetch_factor=conf.training.dataloader.prefetch_factor)

    tensorboard_logger = TensorBoardLogger(save_dir=conf.training.logging.log_dir, name="tensorboard",
                                           version=conf.training.logging.log_version)
    csv_logger = CSVLogger(save_dir=conf.training.logging.log_dir, name="csv",
                           version=conf.training.logging.log_version)

    early_stop_callback = EarlyStopping(
        monitor=conf.training.callbacks.early_stopping.monitor,
        min_delta=0,
        patience=conf.training.callbacks.early_stopping.patience,
        verbose=True,
        mode=conf.training.callbacks.early_stopping.mode,
    )

    checkpoint_callback = ModelCheckpoint(
        filename=conf.training.callbacks.checkpoint.filename,
        auto_insert_metric_name=False,
        save_top_k=conf.training.callbacks.checkpoint.save_top_k,
        monitor=conf.training.callbacks.checkpoint.monitor,
        mode=conf.training.callbacks.checkpoint.mode,
        save_last=True,
        every_n_epochs=1,
    )

    trainer_kwargs = dict(
        precision=conf.training.precision,
        max_epochs=conf.training.max_epochs,
        min_epochs=conf.training.min_epochs,
        accelerator="gpu",
        devices=conf.training.device,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=512 // conf.training.dataloader.batch_size,
        fast_dev_run=fast_run,  # Run only a single batch when debugging the test pipeline.
    )
    gradient_clip_val = _get_optional_float(conf.training.gradient_clip_val)
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    trainer = lightning.Trainer(**trainer_kwargs)

    trainer.test(model, dataloaders=test_loader)
