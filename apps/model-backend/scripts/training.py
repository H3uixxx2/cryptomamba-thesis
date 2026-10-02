import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
from utils import io_tools
import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_v',
        help="Name of the training config under configs/training/.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Logger backend: 'tb' (TensorBoard) or 'wandb'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    parser.add_argument(
        '--save_checkpoints', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )

    parser.add_argument(
        '--skip_test',
        default=False,
        action='store_true',
        help='Do not run the test split after fit.',
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--monitor',
        default='val/rmse',
        help='Validation metric that selects the checkpoint.',
    )

    args = parser.parse_args()
    return args


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)

    model_config.get('params')['logger_type'] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    return model, normalize


if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get('use_volume')

    # DDP + distributed sampling are only valid on a real CUDA box. On a CPU-only host
    # (or MPS) the run falls back to a single-device strategy and a normal sampler.
    use_ddp = (args.accelerator == 'gpu') and torch.cuda.is_available()
    feature_flags = config.get('feature_flags', {}) or {}
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)

    model, normalize = load_model(config, args.logger_type)

    tmp = vars(args)
    tmp.update(config)

    name = config.get('name', args.expname)
    if args.logger_type == 'tb':
        logger = TensorBoardLogger(args.logdir or "logs", name=name)
        logger.log_hyperparams(args)
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    else:
        raise ValueError('Unknown logger type.')

    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=use_ddp,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   window_size=model.window_size,
                                   )
    
    callbacks = []
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor=args.monitor,
            mode="min",
            filename='epoch{epoch}-val-' + args.monitor.split('/')[-1] + '{' + args.monitor + ':.4f}',
            auto_insert_metric_name=False,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    max_epochs = config.get('max_epochs', args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)

    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices,
                         max_epochs=max_epochs,
                         enable_checkpointing=args.save_checkpoints,
                         log_every_n_steps=10,
                         logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False) if use_ddp else 'auto',
                         )

    trainer.fit(model, datamodule=data_module)
    if args.save_checkpoints and not args.skip_test:
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
