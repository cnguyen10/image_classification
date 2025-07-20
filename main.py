import os
from pathlib import Path
import random
from functools import partial

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp

import flax.nnx as nnx
from flax.traverse_util import flatten_dict

import optax

from orbax import checkpoint as ocp

import grain.python as grain

import mlflow

from DataSource import ImageDataSource

from utils import (
    init_tx,
    initialize_dataloader
)
from mixup import mixup_data


@nnx.jit
def cross_entropy_loss(model: nnx.Module, x: jax.Array, y: jax.Array) -> jax.Array:
    """
    """
    logits = model(x)

    loss = optax.losses.softmax_cross_entropy(
        logits=logits,
        labels=y
    )

    loss = jnp.mean(a=loss, axis=0)

    return loss


@nnx.jit
def train_step(x: jax.Array, y: jax.Array, optimizer: nnx.Optimizer):
    """
    """
    grad_value_fn = nnx.value_and_grad(f=cross_entropy_loss, argnums=0)
    loss, grads = grad_value_fn(optimizer.model, x, y)

    optimizer.update(grads=grads)

    return (optimizer, loss)


def train(
    data_loader: grain.DatasetIterator,
    optimizer: nnx.Optimizer,
    cfg: DictConfig
) -> tuple[nnx.Optimizer, jax.Array]:
    """
    """
    # metric to track the training loss
    loss_accum = nnx.metrics.Average()

    # set train mode
    optimizer.model.train()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.train // cfg.training.batch_size),
        desc='train',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(data_loader)

        x = jnp.array(object=samples['image'], dtype=eval(cfg.jax.dtype))
        y = jnp.array(object=samples['label'], dtype=jnp.int32)

        # convert labels to one-hot vectors
        y = jax.nn.one_hot(x=y, num_classes=cfg.dataset.num_classes)

        if cfg.mixup.enable:
            key = jax.random.PRNGKey(seed=optimizer.step.value)
            x_mixed, y = mixup_data(x, y, key, cfg.mixup.beta.a, cfg.mixup.beta.b)
            x = jnp.astype(x_mixed, x.dtype)

        optimizer, loss = train_step(x=x, y=y, optimizer=optimizer)
            

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        loss_accum.update(values=loss)

    return (optimizer, loss_accum.compute())


def evaluate(
    data_loader: grain.DataLoader,
    optimizer: nnx.Optimizer,
    num_samples: int,
    batch_size: int,
    progress_bar_flag: bool,
    dtype: jax.typing.DTypeLike = jnp.float32
) -> jax.Array:
    """
    """
    # metrics for tracking
    acc_accum = nnx.metrics.Accuracy()

    # switch to eval mode
    optimizer.model.eval()

    for samples in tqdm(
        iterable=data_loader,
        desc='eval',
        total=num_samples // batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not progress_bar_flag
    ):
        x = jnp.array(object=samples['image'], dtype=dtype)
        y = jnp.array(object=samples['label'], dtype=jnp.int32)

        logits = optimizer.model(x)

        acc_accum.update(logits=logits, labels=y)
    
    return acc_accum.compute()


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(cfg: DictConfig) -> None:
    """main procedure
    """
    # region ENVIRONMENT
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    # endregion

    # region DATASETS
    source_train = ImageDataSource(
        json_file=cfg.dataset.train_file,
        root=cfg.dataset.root
    )

    source_test = ImageDataSource(
        json_file=cfg.dataset.test_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(source_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(source_test),
        force_add=True
    )
    # endregion
    
    # region MODELS
    model = hydra.utils.instantiate(config=cfg.model)(
        num_classes=cfg.dataset.num_classes,
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100))),
        dropout_rate=cfg.training.dropout_rate,
        dtype=eval(cfg.jax.dtype)
    )

    state = nnx.Optimizer(
        model=model,
        tx=init_tx(
            dataset_length=len(source_train),
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.training.clipped_norm,
            key=random.randint(a=0, b=100)
        )
    )

    del model
    # endregion

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # mlflow.set_system_metrics_sampling_interval(interval=600)
    # mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_id=cfg.experiment.run_id, log_system_metrics=False) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(os.getcwd(), cfg.experiment.logdir, cfg.experiment.name, mlflow_run.info.run_id)

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:

            if cfg.experiment.run_id is None:
                start_epoch_id = 0

                # log hyper-parameters
                mlflow.log_params(
                    params=flatten_dict(xs=OmegaConf.to_container(cfg=cfg), sep='.')
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.StandardRestore(item=nnx.state(state.model))
                )

                nnx.update(state.model, checkpoint)

                del checkpoint
            
            # create iterative datasets as data loaders
            dataloader_train = initialize_dataloader(
                data_source=source_train,
                num_epochs=cfg.training.num_epochs - start_epoch_id + 1,
                shuffle=True,
                seed=random.randint(a=0, b=255),
                batch_size=cfg.training.batch_size,
                resize=cfg.data_augmentation.resize,
                padding_px=cfg.data_augmentation.padding_px,
                crop_size=cfg.data_augmentation.crop_size,
                mean=cfg.data_augmentation.mean,
                std=cfg.data_augmentation.std,
                p_flip=cfg.data_augmentation.prob_random_flip,
                num_workers=cfg.data_loading.num_workers,
                num_threads=cfg.data_loading.num_threads,
                prefetch_size=cfg.data_loading.prefetch_size
            )
            dataloader_train = iter(dataloader_train)

            data_loader_test_fn = partial(
                initialize_dataloader,
                num_epochs=1,
                shuffle=False,
                seed=0,
                batch_size=cfg.training.batch_size,
                resize=cfg.data_augmentation.crop_size,
                padding_px=None,
                crop_size=None,
                mean=cfg.data_augmentation.mean,
                std=cfg.data_augmentation.std,
                p_flip=None,
                is_color_img=True,
                num_workers=cfg.data_loading.num_workers,
                num_threads=cfg.data_loading.num_threads,
                prefetch_size=cfg.data_loading.prefetch_size
            )
            dataloader_test = data_loader_test_fn(source_test)
            dataloader_train_1 = data_loader_test_fn(source_train)

            for epoch_id in tqdm(
                iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not cfg.data_loading.progress_bar
            ):
                state, loss = train(
                    data_loader=dataloader_train,
                    optimizer=state,
                    cfg=cfg
                )

                # wait for checkpoint manager completing the asynchronous saving
                ckpt_mngr.wait_until_finished()

                # save parameters asynchronously
                ckpt_mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.StandardSave(nnx.state(state.model))
                )

                if (epoch_id + 1) % cfg.training.eval_every_n_epochs == 0:
                    accuracy_train = evaluate(
                        data_loader=dataloader_train_1,
                        optimizer=state,
                        num_samples=cfg.dataset.length.train,
                        batch_size=cfg.training.batch_size,
                        progress_bar_flag=cfg.data_loading.progress_bar,
                        dtype=eval(cfg.jax.dtype)
                    )

                    accuracy = evaluate(
                        data_loader=dataloader_test,
                        optimizer=state,
                        num_samples=cfg.dataset.length.test,
                        batch_size=cfg.training.batch_size,
                        progress_bar_flag=cfg.data_loading.progress_bar,
                        dtype=eval(cfg.jax.dtype)
                    )

                    mlflow.log_metrics(
                        metrics={'loss': loss, 'accuracy/test': accuracy, 'accuracy/train': accuracy_train},
                        step=epoch_id + 1,
                        synchronous=False
                    )

    return None


if __name__ == '__main__':
    main()
