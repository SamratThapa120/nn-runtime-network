# Copyright 2023 The tpu_graphs Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to read layout .npz data files to `tf.data.Dataset`.

The high-level function is `get_npz_dataset`, which can be called as:

```
dataset_partitions = get_npz_dataset('~/data/tpugraphs/npz/layout/xla/random')
# Then access: dataset_partitions.{train, vaildation, test}
# You may substite 'xla' with 'nlp' and 'random' with 'default'
```
"""

import collections
import functools
import hashlib
import io
import os
from typing import NamedTuple

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tqdm
import glob
_TOY_DATA = flags.DEFINE_bool(
    'toy_data', False,
    'If set, then only 5 examples will be used in each of '
    '{train, test, validation} partitions.')


class LayoutExample(NamedTuple):
  """Single example of layout graph."""
  total_nodes: tf.Tensor  # shape []
  total_edges: tf.Tensor  # shape []
  total_configs: tf.Tensor  # shape []
  total_config_nodes: tf.Tensor  # shape []

  node_features: tf.Tensor  # shape [total_nodes, node_feat_size]
  node_ops: tf.Tensor  # shape [total_nodes]
  edges: tf.Tensor  # shape [total_edges, 2]
  # shape[total_configs, total_config_nodes, conf_feat_size]:
  node_config_features: tf.Tensor
  config_runtimes: tf.Tensor  # shape [total_configs]
  argsort_config_runtimes: tf.Tensor  # shape [total_configs]
  graph_id: tf.Tensor  # shape []

  node_config_ids: tf.Tensor  # shape [total_config_nodes]
  node_splits: tf.Tensor

  def to_graph_tensor(
      self, config_samples: int = -1, max_nodes: int = -1) -> tfgnn.GraphTensor:
    """Returns `GraphTensor` (sampled if `max(max_nodes, config_samples) >= 0`).

    Args:
      config_samples: if -1, then all module configurations (and their runtimes)
        are returned. If >=0, then this many module configurations (and their
        corresponding runtimes) are sampled uniformly at random.
      max_nodes: Number of nodes to keep in `"sampled_feed"` and
        `"sampled_config"` edge sets. Regardless, edges for all nodes will be
        present in `"feed"` and `"config"`. If `< 0`, then `"sampled_config"`
        and `"config"` will be identical, also `"sampled_feed"` and `"feed"`.

    Returns:
      GraphTensor with node-sets:
        + `"op"` with features=(
            'op': int-vector, 'feats': float-matrix,
            'selected': bool-vector indicating if node has edges in
                        `"sampled_*"` edge-sets).
        + `"nconfigs"` (
            feats='feats': float-tensor with shape
              `[num_configurable_nodes, num_configs, config_feat_dim]`).
        + `"g"` (stands for "graph") has one (root) node connecting to all
          `"op"` and `"nconfigs"`.
          features=('graph_id': filename of graph (without .npz extension),
                    'runtimes': vector of targets with shape `[num_configs]`)
      and edge-sets:
        + 'feed': directed edges connecting op-node to op-node.
        + 'config': edges connecting each `"nconfig"` node with a different
          `"op"` node.
        + 'sampled_feed' and 'sampled_config': contain a subset of edges of the
          above two. Specifically, ones incident on sampled `op` nodes, for
          which feature `selected` is set.
        + 'g_op': edges connecting the singleton `"g"` node to every `"op"` node
        + 'g_config': edges connecting the singleton `"g"` node to every
          `"nconfig"` node.
    """
    config_features = self.node_config_features
    config_runtimes = self.config_runtimes
    num_config_nodes = tf.shape(config_features)[1]
    config_node_ids = tf.range(num_config_nodes, dtype=tf.int32)

    # If sampling is requested.
    if config_samples >= 0:
      argsort_config_runtimes = self.argsort_config_runtimes
      input_num_configs = tf.shape(self.config_runtimes)[0]
      # Skew sampling towards good runtimes.
      select_idx = tf.nn.top_k(
          # Sample wrt GumbulSoftmax([NumConfs, NumConfs-1, ..., 1])
          tf.cast(
              (input_num_configs - tf.range(input_num_configs))
              /input_num_configs, tf.float32) - tf.math.log(
                  -tf.math.log(tf.random.uniform([input_num_configs], 0, 1))),
          config_samples)[1]

      select_idx = tf.gather(argsort_config_runtimes, select_idx)
      # num_configs = config_samples
      config_runtimes = tf.gather(config_runtimes, select_idx)
      config_features = tf.gather(config_features, select_idx)

    ## As we do dropout on (sampled) nodes, maintain a list of edges to keep.
    keep_feed_src = full_feed_src = self.edges[:, 0]
    keep_feed_tgt = full_feed_tgt = self.edges[:, 1]
    keep_config_src = full_config_src = tf.range(
        tf.shape(self.node_config_ids)[0])
    keep_config_tgt = full_config_tgt = tf.cast(
        self.node_config_ids, tf.int32)
    op_node_ids = tf.range(self.total_nodes, dtype=tf.int32)
    node_is_selected = tf.ones([self.total_nodes], dtype=tf.bool)
    kept_node_ratio = tf.ones([], dtype=tf.float32)
    node_ops = self.node_ops
    node_feats = self.node_features

    if max_nodes >= 0:
      num_segments = tf.cast(
          tf.math.ceil(self.total_nodes / max_nodes), tf.int32)
      segment_id = tf.random.uniform(
          shape=[], minval=0, maxval=num_segments, dtype=tf.int32)
      start_idx = segment_id * max_nodes
      end_idx = (segment_id + 1) * max_nodes
      end_idx = tf.minimum(end_idx, self.total_nodes)
      node_is_selected = tf.logical_and(
          op_node_ids >= start_idx, op_node_ids < end_idx)

      feed_edge_mask = tf.logical_and(
          self.edges >= start_idx, self.edges < end_idx)
      feed_edge_mask = tf.logical_and(
          feed_edge_mask[:, 0], feed_edge_mask[:, 1])
      config_edge_mask = tf.logical_and(
          full_config_tgt >= start_idx, full_config_tgt < end_idx)

      kept_node_ratio = tf.cast((end_idx - start_idx) / self.total_nodes,
                                tf.float32)

      keep_feed_src = tf.boolean_mask(full_feed_src, feed_edge_mask)
      keep_feed_tgt = tf.boolean_mask(full_feed_tgt, feed_edge_mask)

      keep_config_src = tf.boolean_mask(full_config_src, config_edge_mask)
      keep_config_tgt = tf.boolean_mask(full_config_tgt, config_edge_mask)

    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            'op': tfgnn.NodeSet.from_fields(
                sizes=tf.shape(op_node_ids),
                features={
                    'op': node_ops,
                    'feats': node_feats,
                    'selected': node_is_selected,
                }
            ),
            'nconfig': tfgnn.NodeSet.from_fields(  # Node-specific configs.
                features={
                    'feats': tf.transpose(config_features, [1, 0, 2]),
                },
                sizes=tf.shape(self.node_config_ids),
            ),
            'g': tfgnn.NodeSet.from_fields(
                features={
                    'graph_id': tf.expand_dims(self.graph_id, 0),
                    'runtimes': tf.expand_dims(config_runtimes, 0),
                    'kept_node_ratio': tf.expand_dims(kept_node_ratio, 0),
                },
                sizes=tf.constant([1]))
        },
        edge_sets={
            'config': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(full_config_src),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('nconfig', full_config_src),
                    target=('op', full_config_tgt))),
            'feed': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(full_feed_src),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('op', full_feed_src),
                    target=('op', full_feed_tgt))),
            'g_op': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(op_node_ids),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('g', tf.zeros_like(op_node_ids)),
                    target=('op', op_node_ids))),
            'g_config': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(config_node_ids),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('g', tf.zeros_like(config_node_ids)),
                    target=('nconfig', config_node_ids))),
            'sampled_config': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(keep_config_src),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('nconfig', keep_config_src),
                    target=('op', keep_config_tgt))),
            'sampled_feed': tfgnn.EdgeSet.from_fields(
                sizes=tf.shape(keep_feed_src),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('op', keep_feed_src),
                    target=('op', keep_feed_tgt))),
        })


class NpzDatasetPartition:
  """Holds one data partition (train, test, validation) on device memory."""

  def __init__(self,files, min_configs=2, max_configs=-1,normalizers=None):
    self.files = glob.glob(os.path.join(files,"*.npz"))
    self.min_configs = min_configs
    self.max_configs = max_configs
    self.normalize = normalizers is not None
    if normalizers:
        minf,maxf = tf.convert_to_tensor(normalizers["max_node_feat"]),tf.convert_to_tensor(normalizers["min_node_feat"]),
        self.node_feat_norms = (
          minf != maxf, tf.expand_dims(minf,0), tf.expand_dims(maxf,0)
        )
        minf,maxf = tf.convert_to_tensor(normalizers["max_node_config_feat"]),tf.convert_to_tensor(normalizers["min_node_config_feat"]),
        self.node_config_feat_norms = (
          minf != maxf, tf.expand_dims(tf.expand_dims(minf,0),0), tf.expand_dims(tf.expand_dims(maxf,0),0)
        )
  def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat,axis=1):
    feature_matrix = tf.boolean_mask(feature_matrix, used_columns, axis=axis)
    min_feat = tf.boolean_mask(min_feat, used_columns, axis=axis)
    max_feat = tf.boolean_mask(max_feat, used_columns, axis=axis)
    return (feature_matrix - min_feat) / (max_feat - min_feat)
  
  def get_item(self, index) -> LayoutExample:
    print(index)
    index = index.numpy()
    npz_file = np.load(tf.io.gfile.GFile(self.files[index], 'rb'))
    graph_id = os.path.splitext(os.path.basename(self.files[index]))[0]

    npz_data = dict(npz_file.items())
    num_configs = npz_data['node_config_feat'].shape[0]
    assert npz_data['node_config_feat'].shape[2] == 18
    npz_data['node_splits'] = npz_data['node_splits'].reshape([-1])
    npz_data['argsort_config_runtime'] = np.argsort(npz_data['config_runtime'])
    if num_configs < self.min_configs:
      print('graph has only %i configurations' % num_configs)
    if self.max_configs > 0 and num_configs > self.max_configs:
      third = self.max_configs // 3
      keep_indices = np.concatenate([
          npz_data['argsort_config_runtime'][:third],  # Good configs.
          npz_data['argsort_config_runtime'][-third:],  # Bad configs.
          np.random.choice(
              npz_data['argsort_config_runtime'][third:-third],
              self.max_configs - 2 * third)
      ], axis=0)
      num_configs = self.max_configs
      npz_data['node_config_feat'] = npz_data['node_config_feat'][keep_indices]
      npz_data['config_runtime'] = npz_data['config_runtime'][keep_indices]
    node_feats = tf.convert_to_tensor(npz_file["node_feat"])
    if self.normalize:
      node_feats = self._apply_normalizer(node_feats,*self.node_feat_norms,axis=1)
    node_conf_feats = tf.convert_to_tensor(npz_file["node_config_feat"])
    if self.normalize:
      node_conf_feats = self._apply_normalizer(node_conf_feats,*self.node_config_feat_norms,axis=2)
    return LayoutExample(
        node_features=node_feats,
        node_ops=tf.convert_to_tensor(npz_file["node_opcode"]),
        edges=tf.cast(tf.convert_to_tensor(npz_file["edge_index"]), tf.int32),
        node_config_features=node_conf_feats,
        node_config_ids=tf.cast(tf.convert_to_tensor(npz_file["node_config_ids"]), tf.int32),
        node_splits=tf.convert_to_tensor(npz_file["node_splits"]),
        config_runtimes=tf.convert_to_tensor(npz_file["config_runtime"]),
        argsort_config_runtimes=tf.convert_to_tensor(np.argsort(npz_data['config_runtime'])),
        graph_id=tf.convert_to_tensor(np.array(graph_id)),
        total_nodes=npz_data['node_feat'].shape[0],
        total_edges=npz_data['edge_index'].shape[0],
        total_configs=npz_data['config_runtime'].shape[0],
        total_config_nodes=npz_data['node_config_ids'].shape[0])

  def tf_load_and_preprocess_npy(self,file_path):
    return tf.py_function(self.get_item, [file_path], tf.float32)
  
  def get_graph_tensors_dataset(
      self, config_samples: int, max_nodes: int = -1) -> tf.data.Dataset:
    # if self.edge_ranges is None:
    #   raise ValueError('finalize() was not invoked.')
    dataset = tf.data.Dataset.from_tensor_slices(self.files)
    dataset = dataset.map(self.tf_load_and_preprocess_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        functools.partial(LayoutExample.to_graph_tensor,
                          config_samples=config_samples, max_nodes=max_nodes))
    return dataset

  def iter_graph_tensors(self):
    if self.edge_ranges is None:
      raise ValueError('finalize() was not invoked.')
    assert self.edge_ranges is not None
    for i in range(self.edge_ranges.shape[0] - 1):
      yield self.get_item(i).to_graph_tensor()


class NpzDataset(NamedTuple):
  """Contains all partitions of the dataset."""
  train: NpzDatasetPartition
  validation: NpzDatasetPartition
  test: NpzDatasetPartition

  @property
  def num_ops(self):
    return int(
        tf.reduce_max([
            tf.reduce_max(self.train.node_opcode),
            tf.reduce_max(self.validation.node_opcode),
            tf.reduce_max(self.test.node_opcode),
        ]).numpy()) + 1

  def _get_normalizer(self, feature_matrix) -> tuple:
    max_feat = tf.reduce_max(feature_matrix, axis=0, keepdims=True)
    min_feat = tf.reduce_min(feature_matrix, axis=0, keepdims=True)
    return min_feat[0] != max_feat[0], min_feat, max_feat

  def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat):
    feature_matrix = tf.boolean_mask(feature_matrix, used_columns, axis=1)
    min_feat = tf.boolean_mask(min_feat, used_columns, axis=1)
    max_feat = tf.boolean_mask(max_feat, used_columns, axis=1)
    return (feature_matrix - min_feat) / (max_feat - min_feat)

  def normalize(self):
    """Removes constant features and normalizes remaining onto [0, 1].

    The statistics are computed only from train partition then applied to all
    partitions {train, test, validation}.
    """
    normalizer_args = self._get_normalizer(self.train.node_feat)
    self.train.node_feat = self._apply_normalizer(
        self.train.node_feat, *normalizer_args)
    self.validation.node_feat = self._apply_normalizer(
        self.validation.node_feat, *normalizer_args)
    self.test.node_feat = self._apply_normalizer(
        self.test.node_feat, *normalizer_args)

    normalizer_args = self._get_normalizer(self.train.node_config_feat)
    self.train.node_config_feat = self._apply_normalizer(
        self.train.node_config_feat, *normalizer_args)
    self.validation.node_config_feat = self._apply_normalizer(
        self.validation.node_config_feat, *normalizer_args)
    self.test.node_config_feat = self._apply_normalizer(
        self.test.node_config_feat, *normalizer_args)


def f(
    split_path: str, min_configs=2, max_configs=-1,
    cache_dir=None) -> NpzDatasetPartition:
  """Returns data for a single partition."""
  glob_pattern = os.path.join(split_path, '*.npz')
  files = tf.io.gfile.glob(glob_pattern)
  if not files:
    raise ValueError('No files matched: ' + glob_pattern)
  if _TOY_DATA.value:
    files = files[:5]

  cache_filename = None
  if cache_dir:
    if not tf.io.gfile.exists(cache_dir):
      tf.io.gfile.makedirs(cache_dir)
    filename_hash = hashlib.md5(
        f'{split_path}:{min_configs}:{max_configs}:{_TOY_DATA.value}'.encode()
        ).hexdigest()
    cache_filename = os.path.join(cache_dir, f'{filename_hash}-cache.npz')
    print('dataset cache file: ', cache_filename)

  npz_dataset = NpzDatasetPartition(files,min_configs=min_configs, max_configs=max_configs)
  # if cache_filename and tf.io.gfile.exists(cache_filename):
  #   npz_dataset.load_from_file(cache_filename)
  # else:
  #   for filename in tqdm.tqdm(files):
  #     np_data = np.load(tf.io.gfile.GFile(filename, 'rb'))
  #     graph_id = os.path.splitext(os.path.basename(filename))[0]
  #     npz_dataset.add_npz_file(
  #         graph_id, np_data, min_configs=min_configs, max_configs=max_configs)
  #   npz_dataset.finalize()
  #   if cache_filename:
  #     npz_dataset.save_to_file(cache_filename)

  return npz_dataset


def get_npz_dataset(
    root_path: str, min_train_configs=-1, max_train_configs=-1,
    cache_dir: 'None | str' = None) -> NpzDataset:
  """Returns {train, test, validation} partitions of layout dataset collection.

  All partitions will be normalized: statistics are computed from training set
  partition and applied to all partitions.

  Args:
    root_path: Path where dataset lives. It must have subdirectories 'train',
      'test' and 'valid'.
    min_train_configs: If > 0, then layout examples will be filtered to have at
      least this many configurations (features and runtimes).
    max_train_configs: Training and validation graphs will be truncated to
      include only this many configurations. Set this according to your
      available device memory. If you have lots of memory, you may set to -1,
      to include all configurations for all {train, validation} graphs.
    cache_dir: If given, the many files for each of {train, test, validation}
      will be stored as one file (makes loading faster, for future runs).
  """
  # npz_dataset = NpzDataset(
  #     train=get_npz_split(
  #         os.path.join(root_path, 'train'), cache_dir=cache_dir,
  #         min_configs=min_train_configs, max_configs=max_train_configs),
  #     validation=get_npz_split(
  #         os.path.join(root_path, 'valid'), cache_dir=cache_dir,
  #         min_configs=min_train_configs, max_configs=max_train_configs),
  #     test=get_npz_split(
  #         os.path.join(root_path, 'test'), cache_dir=cache_dir))
  normalizers = np.load(os.path.join(root_path, 'normalizers.npy'),allow_pickle=True).item()
  npz_dataset = NpzDataset(
    train=NpzDatasetPartition(
        os.path.join(root_path, 'train'),
        min_configs=min_train_configs, max_configs=max_train_configs,normalizers=normalizers),
    validation=NpzDatasetPartition(
        os.path.join(root_path, 'valid'),
        min_configs=min_train_configs, max_configs=max_train_configs,normalizers=normalizers),
    test=NpzDatasetPartition(os.path.join(root_path, 'test'),normalizers=normalizers))
  return npz_dataset
