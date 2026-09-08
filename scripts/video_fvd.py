import torch as th
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import json
import tensorflow.compat.v1 as tf
from tqdm import tqdm

# Metrics and evaluation datasets
from improved_diffusion.video_datasets import get_eval_dataset
import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion.test_util import parse_eval_run_identifier, SampleDataset, DecodedDataset, decoded_ground_truth_cache_dir
from improved_diffusion.metrics import mmd

tf.disable_eager_execution() # Required for our FVD computation code


class FVD:
    def __init__(self, batch_size, T, frame_shape):
        self.batch_size = batch_size
        self.vid = tf.placeholder("uint8", [self.batch_size, T, *frame_shape])
        self.vid_feature_vec = fvd.create_id3_embedding(fvd.preprocess(self.vid, (224, 224)), batch_size=self.batch_size)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

    def extract_features(self, vid):
        def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
            # From here: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            pad_size = target_length - array.shape[axis]
            if pad_size <= 0:
                return array
            npad = [(0, 0)] * array.ndim
            npad[axis] = (0, pad_size)
            return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        # vid is expected to have a shape of BxTxCxHxW
        B = vid.shape[0]
        vid = np.moveaxis(vid, 2, 4)  # B, T, H, W, C
        vid = pad_along_axis(vid, target_length=self.batch_size, axis=0)
        features = self.sess.run(self.vid_feature_vec, feed_dict={self.vid: vid})
        features = features[:B]
        return features

    @staticmethod
    def compute_fvd(vid1_features, vid2_features):
        return fvd.fid_features_to_metric(vid1_features, vid2_features)


def extract_features(test_dataset, sample_dataset, T, num_videos, batch_size=16):
    _, C, H, W = sample_dataset[0][0].shape
    fvd_handler = FVD(batch_size=batch_size, T=T, frame_shape=[H, W, C])
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sample_loader = th.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    if len(test_dataset) != num_videos or len(sample_dataset) != num_videos:
        raise ValueError(f"Expected {num_videos} test and sample videos, got {len(test_dataset)} and {len(sample_dataset)}.")
    with tf.Graph().as_default():
        all_test_features = []
        all_pred_features = []
        for (test_batch, _), (sample_batch, _) in zip(tqdm(test_loader), sample_loader):
            scale = lambda x: ((x.numpy()+1)*255/2).astype(np.uint8)  # scale from [-1, 1] to [0, 255]
            test_batch = scale(test_batch)
            sample_batch = scale(sample_batch)
            test_features = fvd_handler.extract_features(test_batch)
            sample_features = fvd_handler.extract_features(sample_batch)
            all_test_features.append(test_features)
            all_pred_features.append(sample_features)
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_pred_features = np.concatenate(all_pred_features, axis=0)
    return all_test_features, all_pred_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, required=True,
                        help="Number of test videos to evaluate on (how many video_sample.py generated).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for extracting video features with the I3D model.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which per-video sample to evaluate: the -<idx> suffix of sample_XXXX-<idx>.npy (video_sample.py --sample_idx).")
    parser.add_argument("--decode_chunk_size", type=int, default=5)
    parser.add_argument("--decode_cache_dir", type=str, default="./tmp/decoded_ground_truth")
    args = parser.parse_args()

    parsed = parse_eval_run_identifier(os.path.basename(args.eval_dir))
    T, obs_length, eval_on_train = parsed["T"], parsed["n_obs"], parsed["eval_on_train"]
    lower_frame_range, upper_frame_range = parsed["lower_frame_range"], parsed["upper_frame_range"]
    eval_dataset_config = parsed["eval_dataset_config"]

    fvd_save_path = Path(args.eval_dir) / f"fvd-{args.num_videos}-{args.sample_idx}.txt"
    kvd_save_path = Path(args.eval_dir) / f"kvd-{args.num_videos}-{args.sample_idx}.txt"
    if fvd_save_path.exists() and kvd_save_path.exists():
        fvd_val = np.loadtxt(fvd_save_path).squeeze()
        kvd_val = np.loadtxt(kvd_save_path).squeeze()
        print(f"FVD and KVD are already computed: {fvd_val}, {kvd_val}")
        sys.exit()

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # Load the ground-truth videos to compare the samples against
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=T, train=eval_on_train, spacing_kwargs=dict(n_data=args.num_videos),
                             eval_dataset_config=eval_dataset_config, frame_range=(lower_frame_range, upper_frame_range))
    test_dataset_full = get_eval_dataset(**eval_dataset_args)
    sample_dataset = SampleDataset(samples_path=(Path(args.eval_dir) / "samples"), sample_idx=args.sample_idx, length=args.num_videos)

    # If the ground truth is in VAE latent space, decode it to pixels (cached on disk by DecodedDataset)
    encoded_test_data = test_dataset_full[0][0].shape != sample_dataset[0][0].shape
    subset_indices = list(range(args.num_videos))
    if encoded_test_data:
        cache_dir = decoded_ground_truth_cache_dir(args.decode_cache_dir, model_args.dataset, T, eval_dataset_config,
                                                   (lower_frame_range, upper_frame_range), eval_on_train)
        test_dataset_full = DecodedDataset(test_dataset_full, cache_dir, args.decode_chunk_size,
                                           pre_decode=True, subset_indices=subset_indices)
    test_dataset = th.utils.data.Subset(
        dataset=test_dataset_full,
        indices=subset_indices,
    )

    test_features, gen_features = extract_features(test_dataset, sample_dataset, T=T, num_videos=args.num_videos, batch_size=args.batch_size)
    fvd_val = FVD.compute_fvd(test_features, gen_features)
    kvd_val = mmd.mmd2_poly(test_features, gen_features)
    np.savetxt(fvd_save_path, np.array([fvd_val]))
    np.savetxt(kvd_save_path, np.array([kvd_val]))
    print(f"FVD: {fvd_val}, KVD: {kvd_val}")
