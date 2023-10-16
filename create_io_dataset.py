import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def process_file(filename,outpath):
    npz_file = np.load(tf.io.gfile.GFile(filename, 'rb'))
    graph_id = os.path.splitext(os.path.basename(filename))[0]
    npz_data = dict(npz_file.items())
    assert npz_data['node_config_feat'].shape[2] == 18
    npz_data['node_splits'] = npz_data['node_splits'].reshape([-1])
    npz_data['argsort_config_runtime'] = np.argsort(npz_data['config_runtime'])
    node_feats = npz_file["node_feat"]
    node_conf_feats = npz_file["node_config_feat"]

    options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=2)
    tfrecord_name = os.path.join(outpath,os.path.basename(filename).replace(".npz",".tfrecords"))
    example = tf.train.Example(features=tf.train.Features(feature={
        'node_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_feats.tobytes()])),
        'node_ops':tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["node_opcode"].tobytes()])),
        'edges':tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["edge_index"].tobytes()])),
        'node_config_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_conf_feats.tobytes()])),
        'node_config_ids':tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["node_config_ids"].tobytes()])),
        'node_splits':tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["node_splits"].tobytes()])),
        "config_runtimes":tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["config_runtime"].tobytes()])),
        "argsort_config_runtimes":tf.train.Feature(bytes_list=tf.train.BytesList(value=[npz_file["config_runtime"].tobytes()])),
        "graph_id":tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(graph_id).tobytes()])),
        })).SerializeToString()
    with tf.io.TFRecordWriter(tfrecord_name, options=options) as file_writer:
        file_writer.write(example)
        file_writer.close()

def create_io_files(load_path,save_path):
    # Ensure the directory for saving normalizers exists
    os.makedirs(save_path, exist_ok=True)

    # Iterate through the files in the "train" directory
    for f in tqdm(glob.glob(os.path.join(load_path, "*.npz"))):
        process_file(f,save_path)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_normalizers.py <LOAD_PATH> <SAVE_PATH>. each path must contain train,test,valid subfolders")
        sys.exit(1)

    load_path = sys.argv[1]
    save_path = sys.argv[2]
    for splits in ["train","valid","test"]:
        if os.path.exists(os.path.join(load_path,splits)):
            print("Creating io files for : ",splits)
        create_io_files(os.path.join(load_path,splits),os.path.join(save_path,splits))
