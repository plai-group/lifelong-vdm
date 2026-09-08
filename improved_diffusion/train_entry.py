"""
Shared entrypoint for the training scripts.

`scripts/video_train.py` (U-Net) and `scripts/video_train_vdt.py` (VDT) differ only in the model
type; the CLI, argument resolution, and training loop launch live here.
"""

import os
import sys
import argparse
import wandb
import torch.distributed as dist

from improved_diffusion import dist_util
from improved_diffusion.video_datasets import (
    load_data,
    default_T_dict,
    default_image_size_dict,
    default_full_image_size_dict,
)
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.logger import logger

# Keys that model_and_diffusion_defaults() registers but resolve_args() always derives from
# --dataset and --diffusion_space. They are kept off the CLI so a user value is never silently ignored.
DERIVED_MODEL_KEYS = {
    "unet": ("image_size", "in_channels", "diffusion_space_kwargs"),
    "vdt": ("input_size", "in_channels", "num_frames", "diffusion_space_kwargs"),
}


def apply_unobserve_flag():
    # `--unobserve` is consumed before argparse runs; it forces offline wandb logging.
    os.environ["MY_WANDB_DIR"] = "none"
    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "offline"
        if "WANDB_DIR_DRYRUN" in os.environ:
            os.environ["MY_WANDB_DIR"] = os.environ["WANDB_DIR_DRYRUN"]


def init_wandb(config, id):
    if dist.get_rank() != 0:
        return
    wandb_dir = os.environ.get("MY_WANDB_DIR", "none")
    if wandb_dir == "none":
        wandb_dir = None
    if "WANDB_ENTITY" not in os.environ or "WANDB_PROJECT" not in os.environ:
        print("WANDB_ENTITY and/or WANDB_PROJECT environment variables are not set. Running wandb in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(entity=os.environ.get('WANDB_ENTITY'),
               project=os.environ.get('WANDB_PROJECT'),
               config=config, dir=wandb_dir, id=id)
    print(f"Wandb run id: {wandb.run.id}")
    num_nodes = 1
    if "SLURM_JOB_NODELIST" in os.environ:
        assert "SLURM_JOB_NUM_NODES" in os.environ
        num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        print(f"Node list: {os.environ['SLURM_JOB_NODELIST']}")
    logger.logkv("num_nodes", num_nodes)
    print(f"Number of nodes: {num_nodes}")


def num_available_cores():
    # Copied from pytorch source code https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest or 1


def resolve_args(args, model_type):
    """Fill in the values derived from --dataset and --diffusion_space, and validate flag combinations."""
    if args.num_workers == -1:
        # Set the number of workers automatically.
        args.num_workers = max(num_available_cores() - 1, 1)
        print(f"num_workers is not specified. It is automatically set to \"number of cores - 1\" = {args.num_workers}")

    # Frames per training window: T comes from the dataset unless given, and max_frames follows T unless given.
    args.T = default_T_dict[args.dataset] if args.T == -1 else args.T
    args.max_frames = args.T if args.max_frames == -1 else args.max_frames
    if args.T != args.max_frames:
        raise ValueError(f"--T ({args.T}) and --max_frames ({args.max_frames}) must be equal.")

    # Model input shape: RGB pixels, or 4-channel Stable Diffusion latents.
    args.in_channels = 3 if args.diffusion_space == "pixel" else 4
    if model_type == "unet":
        args.image_size = default_image_size_dict[args.dataset]
    elif model_type == "vdt":
        args.input_size = default_full_image_size_dict[args.dataset]
        args.num_frames = args.T
    else:
        raise ValueError(f"unsupported model type: {model_type}")
    args.diffusion_space_kwargs = {
        "diffusion_space": args.diffusion_space,
        "pre_encoded": args.diffusion_space == "latent",
    }
    args.model_type = model_type

    if args.dataset.startswith("streaming_"):
        # Unless specified, default to training on only the most recent frames without a replay buffer.
        if args.ltm_size == -1:
            args.ltm_size = 0
            print("ltm_size is not specified. Defaulting to 0 (no replay buffer).")
        if args.n_sample_stm == -1:
            args.n_sample_stm = args.batch_size
            print(f"n_sample_stm is not specified. Defaulting to batch_size ({args.batch_size}).")
        if not (1 <= args.n_sample_stm <= args.batch_size):
            raise ValueError(f"n_sample_stm must be between 1 and batch_size ({args.batch_size}), but got {args.n_sample_stm}.")
        if args.ltm_size == 0 and args.n_sample_stm != args.batch_size:
            raise ValueError("With ltm_size=0 (no replay buffer), n_sample_stm must equal batch_size so every batch slot reads from the live stream.")
        if args.ltm_size > 0 and args.n_sample_stm == args.batch_size:
            raise ValueError(f"ltm_size={args.ltm_size} but n_sample_stm == batch_size ({args.batch_size}): every batch slot reads from the live stream, "
                             "so the replay buffer would never be sampled. Pass --n_sample_stm smaller than batch_size (or --ltm_size=0 for no replay).")
    elif args.ltm_size != -1 or args.n_sample_stm != -1:
        raise ValueError(f"--ltm_size and --n_sample_stm only apply to streaming_ datasets; offline training on {args.dataset} "
                         "samples i.i.d. from the whole dataset and ignores them.")
    return args


def main(model_type):
    apply_unobserve_flag()
    args = resolve_args(create_argparser(model_type).parse_args(), model_type)

    dist_util.setup_dist()
    init_wandb(config=args, id=args.resume_id or None)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(model_type=model_type,
        **args_to_dict(args, model_and_diffusion_defaults(model_type=model_type).keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    print("creating data loader...")
    data = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        T=args.T,
        num_workers=args.num_workers,
        resume_id=args.resume_id,
        seed=args.data_seed,
        buffer_size=args.ltm_size,
        n_sequential=args.n_sample_stm,
        save_every=args.save_interval,
        frame_range=(0, args.upper_frame_range),
        extra_iters=args.extra_iters,
    )

    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        diffusion_space_kwargs=args.diffusion_space_kwargs,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_interval=args.sample_interval,
        pad_with_random_frames=args.pad_with_random_frames,
        max_frames=args.max_frames,
        enc_dec_chunk_size=args.enc_dec_chunk_size,
        steps_per_experience=args.steps_per_experience,
        masking_mode=args.masking_mode,
        clip_grad=args.clip_grad,
        optimizer=args.optimizer,
        args=args,
        distill_weight=args.distill_weight,
    ).run_loop()


def create_argparser(model_type):
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100000,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        resume_id='',  # set this to a previous run's wandb id to resume training
        num_workers=-1,
        pad_with_random_frames=True,
        max_frames=-1,  # defaults to T
        enc_dec_chunk_size=20,
        T=-1,  # defaults to the dataset's window length
        sample_interval=50000,
        ltm_size=-1,  # replay buffer capacity in K-frame windows (window-start indices) for streaming datasets; 0 disables replay
        n_sample_stm=-1,  # number of batch slots filled from the live stream; the rest are replay samples
        steps_per_experience=1,
        masking_mode="autoregressive",
        optimizer="adam",
        data_seed=0,
        extra_iters=0,
        distill_weight=0.0,
    )
    defaults.update(model_and_diffusion_defaults(model_type=model_type))
    for key in DERIVED_MODEL_KEYS[model_type]:
        defaults.pop(key)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(default_T_dict))
    add_dict_to_argparser(parser, defaults)
    # Arguments whose type add_dict_to_argparser cannot infer from a default of None, or that take a fixed set of values.
    parser.add_argument("--diffusion_space", default="pixel", choices=("pixel", "latent"))
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--upper_frame_range", type=int, default=None)
    return parser
