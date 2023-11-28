import copy
from functools import partial
import json
from typing import List, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    action_encoding_length,
    ActionEncoding,
    allocate_threads,
    get_dataset_statistics,
    make_zero_actions,
    normalize_action_and_proprio,
    pprint_data_mixture,
    state_encoding_length,
    StateEncoding,
    tree_map,
)


def _chunk_act_obs(
    traj,
    window_size,
    additional_action_window_size=0,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
):
    """
    Chunks actions and observations into the given window_size.

    The "action" and "observation" keys are each given a new axis (at index 1) of size `window_size`.
    """
    traj_len = tf.shape(traj["action"])[0]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + additional_action_window_size),
        [traj_len, window_size + additional_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + additional_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "tasks" in traj:
        goal_timestep = traj["tasks"]["goal_timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len, dtype=tf.int32)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None] - 1
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Actions past the goal timestep become no-ops
    action_past_goal = action_chunk_indices > goal_timestep[:, None] - 1
    zero_actions = make_zero_actions(traj["action"], action_encoding)
    traj["action"] = tf.where(
        action_past_goal[:, :, None], zero_actions, traj["action"]
    )
    return traj


def _decode_images(obs: dict) -> dict:
    """Decodes images and depth images, marking empty strings as padding."""
    # indicates which keys in the dict are padding
    pad_mask_dict = {}
    for key in obs:
        if "image" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1, 3), dtype=tf.uint8)
                    pad_mask_dict[key] = False
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.uint8
                    )
                    pad_mask_dict[key] = True
            elif obs[key].dtype == tf.uint8:
                pad_mask_dict[key] = True
            else:
                raise ValueError(
                    f"Unsupported image dtype: found {key} with dtype {obs[key].dtype}"
                )
        elif "depth" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1), dtype=tf.float32)
                    pad_mask_dict[key] = False
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.float32
                    )[..., 0]
                    pad_mask_dict[key] = True
            elif obs[key].dtype == tf.float32:
                pad_mask_dict[key] = True
            else:
                raise ValueError(
                    f"Unsupported depth dtype: found {key} with dtype {obs[key].dtype}"
                )

    obs["pad_mask_dict"] = pad_mask_dict
    return obs


def _augment(obs: dict, seed, augment_kwargs) -> dict:
    """Augments images, skipping padding images."""
    for key in obs:
        if "image" in key:
            if obs["pad_mask_dict"][key]:
                obs[key] = dl.transforms.augment_image(
                    obs[key], **augment_kwargs, seed=seed
                )
    return obs


def apply_common_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    image_augment_kwargs: dict = {},
    task_augmentation_strategy: Optional[str] = None,
    task_augmentation_kwargs: dict = {},
    window_size: int = 1,
    additional_action_window_size: int = 0,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
    resize_size: Optional[Tuple[int, int]] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Common transforms shared between all datasets. Takes and returns a dataset of trajectories. Includes
    the most CPU-intensive operations (image decoding, resizing, augmentation, and chunking).

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects augmentation).
        goal_relabeling_strategy (Optional[str], optional): The goal relabeling strategy to use, or None for no goal
            relabeling. See `bc_goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        image_augment_kwargs (dict, optional): Keyword arguments to pass to the augmentation function. See
            `dlimp.augmentations.augment_image` for documentation.
        task_augmentation_strategy (Optional[str], optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augmentation_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
        resize_size (tuple, optional): target (height, width) for all RGB and depth images, default to no resize.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        additional_action_window_size (int, optional): The number of additional actions to include in the chunked actions.
        action_encoding (ActionEncoding): type of action encoding used, e.g. joint delta vs EEF delta.
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    # adds the "tasks" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(bc_goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

        def move_language_instruction_to_tasks(traj):
            traj["tasks"]["language_instruction"] = traj.pop("language_instruction")
            return traj

        dataset = dataset.map(move_language_instruction_to_tasks, num_parallel_calls)

    def truncate(traj):
        traj_len = tf.shape(traj["action"])[0]
        if traj_len > 100:
            start_idx = tf.random.uniform(
                [], minval=0, maxval=traj_len - 100, dtype=tf.int32
            )
            traj = tf.nest.map_structure(lambda x: x[start_idx : start_idx + 100], traj)
        return traj

    dataset = dataset.map(truncate, num_parallel_calls)

    if train and task_augmentation_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(task_augmentation, task_augmentation_strategy),
                **task_augmentation_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks actions and observations
    dataset = dataset.map(
        partial(
            _chunk_act_obs,
            window_size=window_size,
            additional_action_window_size=additional_action_window_size,
            action_encoding=action_encoding,
        ),
        num_parallel_calls,
    )

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    train: bool,
    shuffle: bool = True,
    image_obs_keys: Union[str, List[str]] = [],
    depth_obs_keys: Union[str, List[str]] = [],
    state_obs_keys: Union[str, List[str]] = [],
    state_encoding: StateEncoding = StateEncoding.NONE,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
    ram_budget: Optional[int] = None,
    action_proprio_normalization_type: Optional[str] = None,
    dataset_statistics: Optional[Union[dict, str]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, Optional[dict]]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format (see below). Yields a dataset of trajectories. Does not include CPU-intensive operations.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        shuffle (bool, optional): Whether to shuffle the order of tfrecords.
        image_obs_keys (str, List[str], optional): List of image observation keys to be decoded. Mapped to "image_XXX".
            Inserts padding image for each None key.
        depth_obs_keys (str, List[str], optional): List of depth observation keys to be decoded. Mapped to "depth_XXX".
            Inserts padding image for each None key.
        state_obs_keys (str, List[str], optional): List of low-dim observation keys to be decoded.
            Get concatenated and mapped to "proprio". Inserts 1d padding for each None key.
        state_encoding (StateEncoding): type of state encoding used, e.g. joint angles vs EEF pose.
        action_encoding (ActionEncoding): type of action encoding used, e.g. joint delta vs EEF delta.
        ram_budget (int, optional): limits the RAM used by tf.data.AUTOTUNE, unit: GB, forwarded to AutotuneOptions.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset
            statistics for normalization. If `action_proprio_normalization_type` is "normal", this
            should contain "mean" and "std" keys. If `action_proprio_normalization_type` is "bounds",
            this should contain "min" and "max" keys. May also provide "num_transitions" and
            "num_trajectories" keys for downstream usage (e.g., for `make_interleaved_dataset`). If
            not provided, the statistics will be computed on the fly based on the train split of the
            dataset.
        num_parallel_reads: number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls: number of parallel calls for map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_[0...N]        # RGB image observations
            - depth_[0...N]        # depth image observations
            - proprio              # concatenated low-dim observations
        - action                   # action vector
        - language_instruction     # language instruction string
        - is_last                  # boolean indicator, 1 on last step
        - is_terminal              # boolean indicator, 1 on last step *if not timeout*
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    if ram_budget:
        dataset = dataset.with_ram_budget(ram_budget)

    image_obs_keys = (
        [image_obs_keys] if not isinstance(image_obs_keys, Sequence) else image_obs_keys
    )
    depth_obs_keys = (
        [depth_obs_keys] if not isinstance(depth_obs_keys, Sequence) else depth_obs_keys
    )
    state_obs_keys = (
        [state_obs_keys] if not isinstance(state_obs_keys, Sequence) else state_obs_keys
    )

    def restructure(traj):
        # apply any dataset-specific transforms
        rlds_transform = RLDS_TRAJECTORY_MAP_TRANSFORMS[name]

        # skip None transforms
        if rlds_transform is not None:
            traj = rlds_transform(traj)

        # remove unused keys
        keep_keys = [
            "observation",
            "action",
            "language_instruction",
            "is_terminal",
            "is_last",
            "_traj_index",
        ]
        traj = {k: v for k, v in traj.items() if k in keep_keys}

        # extracts RGB images, depth images and proprio based on provided keys, pad for all None keys
        orig_obs = traj.pop("observation")
        traj_len = tf.shape(traj["action"])[0]
        traj["observation"] = {}
        for i, key in enumerate(image_obs_keys):
            if key is None:
                # pad with empty string
                traj["observation"][f"image_{i}"] = tf.repeat("", traj_len)
            else:
                traj["observation"][f"image_{i}"] = orig_obs[key]
        for i, key in enumerate(depth_obs_keys):
            if key is None:
                # pad with empty string
                traj["observation"][f"depth_{i}"] = tf.repeat("", traj_len)
            else:
                traj["observation"][f"depth_{i}"] = orig_obs[key]
        if state_obs_keys:
            proprio = []
            for key in state_obs_keys:
                if key is None:
                    # pad with zero
                    proprio.append(tf.zeros((traj_len, 1), dtype=tf.float32))
                else:
                    proprio.append(tf.cast(orig_obs[key], tf.float32))
            traj["observation"]["proprio"] = tf.concat(proprio, axis=-1)
            # make sure state encoding has correct length
            if state_encoding != StateEncoding.NONE:
                assert traj["observation"]["proprio"].shape[
                    -1
                ] == state_encoding_length(state_encoding), (
                    f"State encoding {state_encoding} for dataset {name} expects {state_encoding_length(state_encoding)}-dim proprio"
                    f" but got {traj['observation']['proprio'].shape[-1]}."
                )

        # make sure action encoding has correct length
        assert traj["action"].shape[-1] == action_encoding_length(action_encoding), (
            f"Action encoding {action_encoding} for dataset {name} expects {action_encoding_length(action_encoding)}-dim actions"
            f" but got {traj['action'].shape[-1]}."
        )
        traj["action"] = tf.cast(traj["action"], tf.float32)

        # check that all other keys are present
        for key in ["action", "language_instruction", "is_last", "is_terminal"]:
            if key not in traj:
                raise ValueError(f"Key {key} is missing from trajectory: {traj}")

        # add state and action encoding info
        traj["observation"]["state_encoding"] = tf.repeat(state_encoding, traj_len)[
            ..., None
        ]
        traj["observation"]["action_encoding"] = tf.repeat(action_encoding, traj_len)[
            ..., None
        ]

        # add timestep info
        traj["observation"]["timestep"] = tf.range(traj_len) + 1

        # add name of dataset
        traj["dataset_name"] = tf.repeat(name, traj_len)

        return traj

    dataset = dataset.map(restructure, num_parallel_calls)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            builder, state_obs_keys, restructure, RLDS_TRAJECTORY_MAP_TRANSFORMS[name]
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    dataset = dataset.map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    transform_kwargs: dict,
    train: bool,
) -> dl.DLataset:
    """Creates a single dataset from kwargs.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        transform_kwargs: kwargs passed to 'apply_common_transforms'.
        train: whether this is a training or validation dataset.
    """
    dataset_kwargs = copy.deepcopy(dataset_kwargs)
    transform_kwargs = copy.deepcopy(transform_kwargs)

    # SPECIAL CASE: if `num_parallel_calls` is not in `transform_kwargs`, use
    # same value as in `dataset_kwargs`
    if "num_parallel_calls" not in transform_kwargs:
        transform_kwargs["num_parallel_calls"] = dataset_kwargs.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )

    dataset, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
    dataset = apply_common_transforms(dataset, **transform_kwargs, train=train)

    def apply_image_transform(frame, fn):
        # tasks is not chunked -- apply fn directly
        frame["tasks"] = fn(frame["tasks"])
        # observation is chunked -- apply fn vmapped
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # decode images (and depth images), marking empty strings as padding
    dataset = dataset.frame_map(
        partial(apply_image_transform, fn=_decode_images),
    )

    # resize images, if requested
    if resize_size := transform_kwargs.get("resize_size", None):
        dataset = dataset.frame_map(
            partial(
                apply_image_transform,
                fn=partial(dl.transforms.resize_images, size=resize_size),
            ),
        )

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    common_dataset_kwargs: dict,
    dataset_kwargs_list: List[dict],
    transform_kwargs: dict,
    train: bool,
    sample_weights: Optional[List[float]] = None,
    shared_threads: Optional[int] = None,
    shared_read_threads: Optional[int] = None,
    global_threads: Optional[int] = None,
    balance_weights: bool = True,
    shuffle_buffer_size: int = 10000,
    batch_size: int = 64,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs.

    Args:
        common_dataset_kwargs: shared arguments for `make_dataset_from_rlds` that are common to all datasets.
            Will override kwargs from `dataset_kwargs_list`.
        dataset_kwargs_list: list of kwargs, each element is passed to `make_dataset_from_rlds` for a single dataset.
        transform_kwargs: kwargs passed to 'apply_common_transforms'.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        balance_weights: if True, the sample weights are multiplied by the number of transitions in
            each dataset. This makes it so that, if all the sample weights are equal, one full iteration
            through the interleaved dataset will correspond to one full iteration through each
            individual dataset (only in expectation, since in practice the sampling is random).
        shuffle_buffer_size: size of the dataset shuffle buffer for interleaved dataset.
    """
    common_dataset_kwargs = copy.deepcopy(common_dataset_kwargs)
    dataset_kwargs_list = copy.deepcopy(dataset_kwargs_list)
    transform_kwargs = copy.deepcopy(transform_kwargs)
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        kwargs = copy.deepcopy(dataset_kwargs)
        kwargs.update(**common_dataset_kwargs)
        _, dataset_statistics = make_dataset_from_rlds(**kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics.append(dataset_statistics)

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    if shared_threads is None:
        threads_per_dataset = [tf.data.AUTOTUNE] * len(dataset_kwargs_list)
    else:
        threads_per_dataset = allocate_threads(shared_threads, sample_weights)
    if shared_read_threads is None:
        reads_per_dataset = [tf.data.AUTOTUNE] * len(dataset_kwargs_list)
    else:
        reads_per_dataset = allocate_threads(shared_read_threads, sample_weights)

    # construct datasets
    datasets = []
    action_encodings = []
    for dataset_kwargs, dataset_statistics, threads, reads in zip(
        dataset_kwargs_list,
        all_dataset_statistics,
        threads_per_dataset,
        reads_per_dataset,
    ):
        kwargs = copy.deepcopy(dataset_kwargs)
        kwargs.update(**common_dataset_kwargs)
        dataset, _ = make_dataset_from_rlds(
            **kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=dataset_statistics,
        )
        action_encodings.append(
            dataset_kwargs.get("action_encoding", ActionEncoding.EEF_POS)
        )
        datasets.append(dataset.repeat())

    # TODO: support interleaving datasets with different action encodings
    assert (
        len(set(action_encodings)) == 1
    ), f"Need action encodings of all datasets to be identical, currently used encodings: {action_encodings}."

    transformed_datasets = []
    for dataset, threads in zip(datasets, threads_per_dataset):
        # only does relabeling and chunking
        transformed_datasets.append(
            apply_common_transforms(
                dataset,
                **transform_kwargs,
                num_parallel_calls=threads,
                train=train,
            ).flatten(num_parallel_calls=threads)
        )

    # interleave at the transition level
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        transformed_datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    def apply_image_transform(frame, fn):
        # tasks is not chunked -- apply fn directly
        frame["tasks"] = fn(frame["tasks"])
        # observation is chunked -- apply fn vmapped
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # decode images (and depth images), marking empty strings as padding
    dataset = dataset.map(
        partial(apply_image_transform, fn=_decode_images), global_threads
    )

    # resize images, if requested
    if resize_size := transform_kwargs.get("resize_size", None):
        dataset = dataset.map(
            partial(
                apply_image_transform,
                fn=partial(dl.transforms.resize_images, size=resize_size),
            ),
            global_threads,
        )

    if train:
        # augment each frame with the same seed
        def aug(frame):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            return apply_image_transform(
                frame,
                lambda x: _augment(
                    x, seed, augment_kwargs=transform_kwargs["image_augment_kwargs"]
                ),
            )

        dataset = dataset.map(aug, global_threads)

    dataset = dataset.batch(batch_size, num_parallel_calls=global_threads)
    return dataset
