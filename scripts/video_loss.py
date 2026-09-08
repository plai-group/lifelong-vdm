"""
Compute the average diffusion loss of a model checkpoint on evaluation video subsequences.

Sample Command
python scripts/video_loss.py checkpoints/p9lrebju/ema_0.9999_050000.pt --eval_dir results/p9lrebju/ema_0.9999_050000_heun-80-inf-0-1-1000-0.002-7-100 --T=10 --stop_index=200 --num_sampled_videos=1000 --max_frames=10 --n_obs=5 --batch_size=25 --trials=10
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch as th

from improved_diffusion import dist_util
from improved_diffusion.script_util import str2bool
from improved_diffusion.test_util import load_model_from_checkpoint
from improved_diffusion.sampling_schemes import sampling_schemes
from improved_diffusion.video_datasets import get_eval_dataset, eval_dataset_configs
from improved_diffusion.resample import create_named_schedule_sampler


def compute_loss(batch, args, model, diffusion, schedule_sampler, trials=10):
    B, T, *_ = batch.shape
    batch = batch.to(dist_util.dev())

    def get_sub_batch_kwargs_iterator(batch):
        frame_indices_iterator = iter(sampling_schemes['autoreg'](
            video_length=T, num_obs=args.n_obs,
            max_frames=args.max_frames, step_size=args.max_latent_frames,
        ))
        frame_indices_iterator.set_videos(batch)  # non-adaptive schemes ignore the video contents but still take the batch size from it
        for obs_frame_indices, latent_frame_indices in frame_indices_iterator:
            frame_indices = th.cat([th.tensor(obs_frame_indices), th.tensor(latent_frame_indices)], dim=1).long()
            x0 = th.stack([batch[i, fi] for i, fi in enumerate(frame_indices)], dim=0).clone()
            obs_mask = th.cat([
                    th.ones_like(th.tensor(obs_frame_indices)),
                    th.zeros_like(th.tensor(latent_frame_indices))],
                dim=1).view(B, -1, 1, 1, 1).float()
            latent_mask = 1 - obs_mask
            obs_mask, latent_mask, frame_indices = map(lambda t: t.to(x0.device), [obs_mask, latent_mask, frame_indices])
            yield dict(
                frame_indices=frame_indices, obs_mask=obs_mask,
                latent_mask=latent_mask, x0=x0
            )
    loss_parts = []
    n_latent_parts = []
    for model_kwargs in get_sub_batch_kwargs_iterator(batch):
        x0 = model_kwargs['x0']
        latent_mask = model_kwargs['latent_mask']
        loss = []
        for _ in range(trials):
            t, weights = schedule_sampler.sample(x0.shape[0], dist_util.dev())
            loss_trial = diffusion.training_losses(
                model, x0, t, model_kwargs=model_kwargs,
                latent_mask=latent_mask, eval_mask=latent_mask)['loss']
            loss_trial *= weights
            loss.append(loss_trial)
        loss = th.stack(loss).mean(dim=0)
        n_latents = latent_mask.view(latent_mask.shape[:2]).sum(dim=-1)
        loss_parts.append((loss * n_latents).cpu())
        n_latent_parts.append(n_latents.cpu())
    loss_sum = th.stack(loss_parts, dim=0).sum(dim=0)
    cnt = th.stack(n_latent_parts, dim=0).sum(dim=0)
    return loss_sum / cnt


@th.no_grad()
def main(args):
    loss_save_path = Path(args.eval_dir) / f"loss-{args.trials}-{args.seed}.txt"
    if loss_save_path.exists():
        loss = np.loadtxt(loss_save_path).squeeze()
        print(f"Losses are already computed: {loss}")
        return
    loss_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.indices = list(range(args.start_index, args.stop_index))
    if args.num_sampled_videos is None:
        args.num_sampled_videos = len(args.indices)
    print(f"Computing loss for indices {args.start_index} to {args.stop_index}.")

    # Build the model and diffusion from the checkpoint (eval mode) and recover its training config
    model, diffusion, model_args = load_model_from_checkpoint(
        args.checkpoint_path, args.device, timestep_respacing=args.timestep_respacing)
    schedule_sampler = create_named_schedule_sampler(model_args.schedule_sampler, diffusion)
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    if args.max_latent_frames is None:
        args.max_latent_frames = args.max_frames // 2

    # Load the evaluation videos (first n_obs frames are observed, the rest are scored)
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=args.T, train=args.eval_on_train,
                             eval_dataset_config=args.eval_dataset_config, spacing_kwargs=dict(n_data=args.num_sampled_videos),
                             frame_range=(args.lower_frame_range, args.upper_frame_range))
    dataset = get_eval_dataset(**eval_dataset_args)
    dataset = th.utils.data.Subset(dataset=dataset, indices=args.indices)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    avg_loss = []
    for batch, _ in tqdm(dataloader):
        loss = compute_loss(
            batch=batch,
            args=args,
            model=model,
            diffusion=diffusion,
            schedule_sampler=schedule_sampler,
            trials=args.trials,
        )
        avg_loss.append(loss)
    avg_loss = th.cat(avg_loss, dim=0).mean().item()
    print(avg_loss)
    np.savetxt(loss_save_path, np.array([avg_loss]))


def create_sampling_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, required=True)
    parser.add_argument("--num_sampled_videos", type=int, default=None,
                        help="Total number of videos in the spaced eval dataset; must match the value given to video_sample.py (default: stop_index-start_index)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--n_obs", type=int, required=True, help="Number of observed frames at the beginning of the video. The diffusion loss is evaluated on the rest.")
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. Defaults to the training dataset's default length (video_datasets.default_T_dict).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Denoted K in the paper. Maximum number of (observed or latent) frames input to the model at once. Defaults to what the model was trained with.")
    parser.add_argument("--max_latent_frames", type=int, default=None, help="Number of latent frames scored in each stage. Defaults to max_frames/2.")
    parser.add_argument("--eval_on_train", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="Tag for the output filename (loss-<trials>-<seed>.txt); does not seed any RNG.")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")

    parser.add_argument("--eval_dataset_config", type=str, default=eval_dataset_configs["default"], choices=list(eval_dataset_configs.keys()))
    parser.add_argument("--lower_frame_range", type=int, default=0, help="Lower bound of the frame index range the eval dataset draws from.")
    parser.add_argument("--upper_frame_range", type=int, default=None, help="Upper bound (exclusive) of the frame index range the eval dataset draws from. None: end of stream.")
    parser.add_argument("--decode_chunk_size", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = create_sampling_parser()
    args = parser.parse_args()
    main(args)
