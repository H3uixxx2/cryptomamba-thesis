import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import yaml
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_dict_sha256(state_dict):
    """Stable content hash used to prove paired arms share initialization."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

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
        help="Logging directory.",
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
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
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
        '--resume_from_checkpoint',
        default=None,
    )

    parser.add_argument(
        '--skip_test',
        default=False,
        action='store_true',
        help='Do not run the test split after fit (smoke runs must stay val-only).',
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--monitor',
        default='val/rmse',
        help='Metric that drives checkpoint selection and early stopping. Distributional '
             'runs must select on val/crps — picking a distributional model by RMSE would '
             'ignore the only part of it that carries signal.',
    )

    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=0,
        help='Stop when val/rmse has not improved for this many epochs (0 = off, the '
             'behaviour of every run recorded so far). Lets the data pick the budget '
             'instead of a hand-set epoch count; ModelCheckpoint still keeps the best '
             'epoch, so stopping early never discards the selected weights.',
    )

    parser.add_argument(
        '--run_manifest',
        default=None,
        help='Optional immutable producer manifest for registered training.',
    )
    parser.add_argument('--preregistration', default=None)
    parser.add_argument('--expected_preregistration_sha256', default=None)

    args = parser.parse_args()
    return args


def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = log_dir + '/hparams.yaml'
    if os.path.exists(path):
        return
    with open(path, 'w') as f:
        yaml.dump(save_dict, f)


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


def initialize_shared_state(target, source):
    """Copy control initialization into a wrapped candidate model.

    The candidate's timescale-initialized ``A_log`` tensors and all
    architecture-only parameters remain untouched.
    """
    target_state = target.state_dict()
    updated_state = dict(target_state)
    copied = []
    preserved_treatment = []

    for source_name, source_value in source.state_dict().items():
        parts = source_name.split('.')
        if len(parts) > 3 and parts[:2] == ['model', 'blocks']:
            target_name = '.'.join(parts[:3] + ['block'] + parts[3:])
        else:
            target_name = source_name

        if source_name.endswith('.op.A_log'):
            if target_name not in target_state:
                raise KeyError(f'Missing candidate timescale tensor: {target_name}')
            preserved_treatment.append(target_name)
            continue

        if target_name not in target_state:
            raise KeyError(f'Missing shared candidate tensor: {target_name}')
        if target_state[target_name].shape != source_value.shape:
            raise ValueError(
                f'Shape mismatch for {source_name} -> {target_name}: '
                f'{tuple(source_value.shape)} != {tuple(target_state[target_name].shape)}'
            )
        updated_state[target_name] = source_value.detach().clone()
        copied.append(target_name)

    target.load_state_dict(updated_state, strict=True)
    return {
        'copied': copied,
        'preserved_treatment': preserved_treatment,
    }


if __name__ == "__main__":

    args = get_args()
    if args.run_manifest:
        manifest_path = Path(args.run_manifest).resolve()
        if manifest_path.exists():
            raise FileExistsError(f'refusing to overwrite run manifest: {manifest_path}')
        if not args.save_checkpoints or not args.skip_test:
            raise ValueError(
                'registered runs require --save_checkpoints and --skip_test'
            )
        if not args.logdir or not args.preregistration or not args.expected_preregistration_sha256:
            raise ValueError(
                'registered runs require logdir and hash-pinned preregistration'
            )
        preregistration_path = Path(args.preregistration).resolve()
        actual_preregistration_sha256 = file_sha256(preregistration_path)
        if actual_preregistration_sha256 != args.expected_preregistration_sha256:
            raise ValueError('training preregistration SHA-256 mismatch')
        preregistration = json.loads(preregistration_path.read_text(encoding='utf-8'))
        for section in ('source_sha256', 'input_sha256'):
            for relative, expected in preregistration.get(section, {}).items():
                registered_path = Path(ROOT) / relative
                if not registered_path.is_file() or file_sha256(registered_path) != expected:
                    raise ValueError(
                        f'training input changed after preregistration: {relative}'
                    )
    else:
        manifest_path = None
        preregistration = None
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    if manifest_path is not None:
        registered_training = preregistration['static_contract']['training']
        registered_arm = registered_training['seed_23_arms'].get(args.config)
        actual_registered_contract = {
            'monitor': args.monitor,
            'loss': config['hyperparams']['loss'],
            'batch_size': args.batch_size,
            'max_epochs': config.get('max_epochs', args.max_epochs),
            'early_stop_patience': args.early_stop_patience,
            'accelerator': args.accelerator,
            'devices': args.devices,
            'num_workers': args.num_workers,
            'skip_test': args.skip_test,
            'save_checkpoints': args.save_checkpoints,
        }
        expected_registered_contract = {
            'monitor': registered_arm['monitor'] if registered_arm else None,
            'loss': registered_arm['loss'] if registered_arm else None,
            'batch_size': registered_training['batch_size'],
            'max_epochs': registered_training['max_epochs'],
            'early_stop_patience': registered_training['early_stop_patience'],
            'accelerator': registered_training['accelerator'],
            'devices': registered_training['devices'],
            'num_workers': registered_training['num_workers'],
            'skip_test': True,
            'save_checkpoints': True,
        }
        if args.seed != 23 or actual_registered_contract != expected_registered_contract:
            raise ValueError('training command does not match preregistered seed-23 contract')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get('use_volume')

    # DDP + distributed sampling are only valid on a real CUDA box (Colab/Linux GPU).
    # On local CPU/MPS (no CUDA) fall back to single-device strategy + a normal sampler
    # so baselines can train locally. The CUDA/Colab path is unchanged.
    use_ddp = (args.accelerator == 'gpu') and torch.cuda.is_available()
    feature_flags = config.get('feature_flags', {}) or {}
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []), **feature_flags)

    model, normalize = load_model(config, args.logger_type)
    shared_init_config = config.get('shared_init_config')
    if shared_init_config:
        cuda_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            pl.seed_everything(args.seed)
            source_config = io_tools.load_config_from_yaml(
                f'{ROOT}/configs/training/{shared_init_config}.yaml'
            )
            source_model, _ = load_model(source_config, args.logger_type)
            init_report = initialize_shared_state(model, source_model)
        del source_model
        print(
            'Shared initialization: '
            f"{len(init_report['copied'])} tensors copied; "
            f"{len(init_report['preserved_treatment'])} treatment timescales preserved."
        )

    if config.get('reset_training_rng_after_init', False):
        pl.seed_everything(args.seed)

    initial_state_hash = state_dict_sha256(model.state_dict())

    tmp = vars(args)
    tmp.update(config)

    name = config.get('name', args.expname)
    if args.logger_type == 'tb':
        logger = TensorBoardLogger(
            args.logdir or "logs",
            name=name,
            version=(f'seed{args.seed}' if manifest_path is not None else None),
        )
        if manifest_path is not None:
            log_path = Path(logger.log_dir)
            if log_path.exists() and any(log_path.iterdir()):
                raise FileExistsError(
                    f'refusing to reuse non-empty registered log directory: {log_path}'
                )
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
                                   cross_boundary_validation=config.get(
                                       'cross_boundary_validation', False),
                                   validation_selection_rows=config.get(
                                       'validation_selection_rows'),
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

    if args.early_stop_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor=args.monitor,
            mode='min',
            patience=args.early_stop_patience,
            verbose=True,
        ))

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
    if manifest_path is not None:
        measured_provenance = {}
        for section in ('source_sha256', 'input_sha256'):
            measured_provenance[section] = {
                relative: file_sha256(Path(ROOT) / relative)
                for relative in preregistration[section]
            }
            if measured_provenance[section] != preregistration[section]:
                raise ValueError(
                    f'{section} changed during registered training; refusing manifest'
                )
        selected = Path(checkpoint_callback.best_model_path).resolve()
        if not selected.is_file():
            raise RuntimeError('registered training did not produce a best checkpoint')
        checkpoint_payload = torch.load(
            selected, map_location='cpu', weights_only=False
        )
        config_path = Path(ROOT) / 'configs' / 'training' / f'{args.config}.yaml'
        training_contract = {
            'monitor': args.monitor,
            'loss': config['hyperparams']['loss'],
            'batch_size': args.batch_size,
            'max_epochs': max_epochs,
            'early_stop_patience': args.early_stop_patience,
            'accelerator': args.accelerator,
            'devices': args.devices,
            'num_workers': args.num_workers,
            'skip_test': args.skip_test,
            'save_checkpoints': args.save_checkpoints,
        }
        manifest = {
            'schema': 'cryptomamba_training_run_v1',
            'completed': True,
            'arm': args.config,
            'seed': args.seed,
            'preregistration_sha256': actual_preregistration_sha256,
            'training_contract': training_contract,
            'config': {
                'path': str(config_path.resolve()),
                'sha256': file_sha256(config_path),
            },
            'source_sha256': measured_provenance['source_sha256'],
            'input_sha256': measured_provenance['input_sha256'],
            'initial_state_sha256': initial_state_hash,
            'checkpoint': {
                'path': str(selected),
                'sha256': file_sha256(selected),
                'monitor': args.monitor,
                'best_score': float(checkpoint_callback.best_model_score),
                'best_epoch': int(checkpoint_payload['epoch']),
            },
            'runtime': {
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(),
                'torch': torch.__version__,
                'lightning': pl.__version__,
                'platform': platform.platform(),
                'log_dir': logger.log_dir,
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(f'run_manifest: {manifest_path}')
        print(f'run_manifest_sha256: {file_sha256(manifest_path)}')
    if args.save_checkpoints and not args.skip_test:
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
