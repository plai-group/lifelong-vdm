import torch as th
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import json

from videojedi import JEDiMetric

# Evaluation datasets and helpers
from improved_diffusion.video_datasets import get_eval_dataset
from improved_diffusion.test_util import parse_eval_run_identifier, SampleDataset, DecodedDataset, decoded_ground_truth_cache_dir


def compute_jedi(jedi_feature_path, test_dataset, sample_dataset, num_videos, batch_size=16):
    def transform_collate(batch):
        return th.stack([(item[0]+1)/2 for item in batch], dim=0), {}

    truth_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=transform_collate)
    sample_loader = th.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=transform_collate)

    # videojedi auto-downloads the V-JEPA weights (~10GB) into model_dir on first use;
    # keep them outside the repository. Override the location with JEDI_MODEL_DIR.
    model_dir = os.path.expanduser(os.environ.get("JEDI_MODEL_DIR", "~/.cache/videojedi"))
    os.makedirs(model_dir, exist_ok=True)
    jedi = JEDiMetric(feature_path=jedi_feature_path, model_dir=model_dir)
    jedi.load_features(truth_loader, sample_loader, num_samples=num_videos)
    jedi_val = jedi.compute_metric()
    return jedi_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, required=True,
                        help="Number of test videos to evaluate on (how many video_sample.py generated).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for extracting video features with the V-JEPA model.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which per-video sample to evaluate: the -<idx> suffix of sample_XXXX-<idx>.npy (video_sample.py --sample_idx).")
    parser.add_argument("--decode_chunk_size", type=int, default=5)
    parser.add_argument("--decode_cache_dir", type=str, default="./tmp/decoded_ground_truth")
    args = parser.parse_args()

    parsed = parse_eval_run_identifier(os.path.basename(args.eval_dir))
    T, obs_length, eval_on_train = parsed["T"], parsed["n_obs"], parsed["eval_on_train"]
    lower_frame_range, upper_frame_range = parsed["lower_frame_range"], parsed["upper_frame_range"]
    eval_dataset_config = parsed["eval_dataset_config"]

    jedi_save_path = Path(args.eval_dir) / f"jedi-{args.num_videos}-{args.sample_idx}.txt"
    if jedi_save_path.exists():
        jedi = np.loadtxt(jedi_save_path).squeeze()
        print(f"JEDi is already computed: {jedi}")
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

    jedi_feature_path = Path(args.eval_dir) / f"jedi_features_{args.sample_idx}"
    jedi_feature_path.mkdir(parents=True, exist_ok=True)

    jedi_val = compute_jedi(jedi_feature_path, test_dataset, sample_dataset, num_videos=args.num_videos, batch_size=args.batch_size)
    np.savetxt(jedi_save_path, np.array([jedi_val]))
    print(f"JEDI: {jedi_val}")
