import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.tuner import Tuner
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data.dataset_openfwi import OpenFWI


def _get_log_every_n_steps(conf, batch_size):
    # Respect explicit config override if present; Lightning requires >= 1.
    if hasattr(conf, "training") and hasattr(conf.training, "log_every_n_steps"):
        try:
            override = int(conf.training.log_every_n_steps)
            return max(1, override)
        except (TypeError, ValueError):
            return 1
    if not batch_size:
        return 1
    return max(1, 512 // int(batch_size))


def _get_persistent_workers(conf, num_workers):
    # persistent_workers only valid when num_workers > 0.
    if num_workers <= 0:
        return False
    if hasattr(conf, "training") and hasattr(conf.training, "dataloader") and hasattr(conf.training.dataloader,
                                                                                      "persistent_workers"):
        return bool(conf.training.dataloader.persistent_workers)
    return True


def base_train(model, conf, fast_run=True, use_lr_finder=False, ckpt_path=None):
    train_dataset_list = []
    val_dataset_list = []
    for dataset in conf.datasets.dataset_name:
        dataset = OpenFWI(root_dir='data/openfwi', use_data=conf.datasets.use_data, datasets=(dataset,),
                          use_normalize=conf.datasets.use_normalize)
        total_size = len(dataset)
        # Reserve last 10% for potential test split (not used here).
        test_size = int(0.1 * total_size)  # Take the last 10% as the test set (non-random)
        remaining_idx = list(range(total_size - test_size))

        # The training and validation sets were randomly split from the remaining data
        train_idx, val_idx = train_test_split(remaining_idx, test_size=0.25,  # Relative to 25% of the remaining data
                                              random_state=42, shuffle=True
                                              # The training and validation sets were randomly partitioned
                                              )
        train_dataset_list.append(Subset(dataset, train_idx))
        val_dataset_list.append(Subset(dataset, val_idx))

    train_set = ConcatDataset(train_dataset_list)
    val_set = ConcatDataset(val_dataset_list)
    batch_size = conf.training.dataloader.batch_size
    num_workers = conf.training.dataloader.num_workers
    persistent_workers = _get_persistent_workers(conf, num_workers)
    log_every_n_steps = _get_log_every_n_steps(conf, batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=persistent_workers, pin_memory=True,
                              prefetch_factor=conf.training.dataloader.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            persistent_workers=persistent_workers, pin_memory=True,
                            prefetch_factor=conf.training.dataloader.prefetch_factor)

    tensorboard_logger = TensorBoardLogger(save_dir=conf.training.logging.log_dir, name='tensorboard',
                                           version=conf.training.logging.log_version, )
    csv_logger = CSVLogger(save_dir=conf.training.logging.log_dir, name="csv",
                           version=conf.training.logging.log_version, )

    early_stop_callback = EarlyStopping(monitor=conf.training.callbacks.early_stopping.monitor, min_delta=0,
                                        patience=conf.training.callbacks.early_stopping.patience, verbose=True,
                                        mode=conf.training.callbacks.early_stopping.mode)

    checkpoint_callback = ModelCheckpoint(  # dirpath='checkpoints',
        filename=conf.training.callbacks.checkpoint.filename, auto_insert_metric_name=False,
        save_top_k=conf.training.callbacks.checkpoint.save_top_k, monitor=conf.training.callbacks.checkpoint.monitor,
        mode=conf.training.callbacks.checkpoint.mode, save_last=True, every_n_epochs=1, )

    # Trainer config: gradient_clip_val is config-driven; None means no clipping.
    trainer_kwargs = dict(precision=conf.training.precision, max_epochs=conf.training.max_epochs,
                          min_epochs=conf.training.min_epochs, accelerator="gpu",  # strategy='ddp_spawn',
                          devices=conf.training.device, logger=[tensorboard_logger, csv_logger],
                          callbacks=[early_stop_callback, checkpoint_callback], log_every_n_steps=log_every_n_steps,
                          # must be >= 1
                          fast_dev_run=fast_run, )
    if conf.training.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = conf.training.gradient_clip_val
    trainer = lightning.Trainer(**trainer_kwargs)

    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(lr_finder.suggestion())
    else:
        if ckpt_path is not None:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader)
