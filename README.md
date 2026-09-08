# Lifelong Learning of Video Diffusion Models From a Single Video Stream

This repository contains the datasets and code for **"Lifelong Learning of Video Diffusion Models From a Single Video Stream"**
(ECCV 2026 Workshop on How to Build Effective World Models for Embodied AI).
[arXiv](https://arxiv.org/abs/2406.04814) · [OpenReview](https://openreview.net/forum?id=oW6PbzOHHa) · [Datasets on the Hugging Face Hub](https://huggingface.co/jason-yoo-108)

Each dataset is one continuous video stream whose training split is one million consecutive frames.
The code supports training autoregressive video diffusion models in an offline fashion (on randomly sampled short video subsequences) and in a lifelong fashion (sliding window of video subsequences from beginning to end).

## Datasets

The table below presents these datasets from the simplest dataset to the most complex.
Four of the five are on the Hugging Face Hub in the directory layout the loaders expect.
We can't redistribute the source videos for Lifelong 3D Maze, so that one is built locally with a script.

| Dataset | `--dataset` | Train / test frames | Model input | Download |
|---|---|---|---|---|
| Lifelong Bouncing Balls (O) | `ball_stn` | 1M / 1M | 32x32x3 pixels, 10 fps | [jason-yoo-108/lifelong-bouncing-balls-o](https://huggingface.co/datasets/jason-yoo-108/lifelong-bouncing-balls-o) |
| Lifelong Bouncing Balls (C) | `ball_nstn` | 1M / 1M | 32x32x3 pixels, 10 fps | [jason-yoo-108/lifelong-bouncing-balls-c](https://huggingface.co/datasets/jason-yoo-108/lifelong-bouncing-balls-c) |
| Lifelong 3D Maze | `wmaze` | 1M / 100k | 64x64x3 pixels, 20 fps | not hosted; built by `datasets/preprocess_wmaze.sh` (see below) |
| Lifelong Drive | `drive` | 1M / 100k | 64x64x4 latents (512x512 source), 20 fps | [jason-yoo-108/lifelong-drive](https://huggingface.co/datasets/jason-yoo-108/lifelong-drive) |
| Lifelong PLAICraft | `plaicraft` | 1M / 500k | 160x96x4 latents (1280x768 source), 10 fps | [jason-yoo-108/lifelong-plaicraft](https://huggingface.co/datasets/jason-yoo-108/lifelong-plaicraft) |

Drive and PLAICraft are stored as SDXL VAE latents rather than pixels. Each dataset card on the Hub
describes the file format and has a short loading snippet, so the data can be used without this repository.

### Download

The download script only needs `huggingface_hub` (refer to the Installation section).

```bash
python datasets/download.py ball_stn drive          # -> datasets/ball_stn, datasets/drive
python datasets/download.py --all                   # the four hosted datasets, ≈172 GB
python datasets/download.py drive --split test --no_mp4   # one split only; --no_mp4 skips Drive's 512x512 source video and keeps the latents
```

The code's dataloaders read datasets placed at `datasets/<name>` relative to the repository root.

#### Notes on Lifelong 3D Maze

`datasets/preprocess_wmaze.sh` downloads the two 10-hour Windows 3D Maze YouTube videos with `yt-dlp`, then crops, rescales and concatenates them into a 64x64, 20 fps stream with `ffmpeg`. It writes the train and test streams as 500-frame npy chunks.

```bash
bash datasets/preprocess_wmaze.sh        # needs a current yt-dlp, ffmpeg, opencv-python; ~40 GB scratch (freed at the end) + 136 GB output
python datasets/verify_wmaze.py          # compares your rebuild with the paper's data; YouTube re-encodes, so expect small differences
```

## Code

The code trains autoregressive video diffusion models with U-Net or VDT backbones under three regimes: offline training, lifelong learning with an experience-replay buffer (ER), and lifelong learning with no replay.
It builds on [flexible-video-diffusion-modeling](https://github.com/plai-group/flexible-video-diffusion-modeling) (FDM), which in turn builds on OpenAI's [improved-diffusion](https://github.com/openai/improved-diffusion).

### Installation

```bash
conda create -n lifelong-vdm python=3.10 -y
conda activate lifelong-vdm
conda install mpich=3.3.2 mpi4py=3.1.4 -y
pip install -r requirements.txt
pip install -e .
```

`mpi4py` is installed through conda rather than pip so that it comes bundled with a matching MPI runtime.
<!-- The MPICH 3.3.2 pin is deliberate: it initializes cleanly as a single process under SLURM's `srun`, whereas newer MPICH builds and conda's library-less `external_*` stubs did not work for us. -->

Run all commands from the repository root.

**JEDi metric only:** `scripts/video_jedi.py` needs a *separate* environment because
`videojedi`'s torch stack conflicts with the TensorFlow version used for FVD:

```bash
conda create -n jedi python=3.10 -y
conda activate jedi
pip install -r requirements-jedi.txt
```

### Training

There are two training scripts, `scripts/video_train.py` for the U-Net and `scripts/video_train_vdt.py` for VDT.
They take the same flags, except the VDT script also needs `--model_name` (`VDT-S`, `VDT-SM` or `VDT-M`) and `--patch_size`.
`--dataset=<name>` samples windows i.i.d. from the whole stream.
`--dataset=streaming_<name>` reads the
same files but walks through them in order, one frame per step. The replay-buffer flags then decide whether the model also gets a buffer:

| Regime | Description | Flags |
|---|---|---|
| Offline Learning | i.i.d. sampling of video windows from the full dataset (standard training) | `--dataset=<name>` (optionally `--extra_iters=N` to continue past one epoch) |
| Lifelong Learning (ER) | in-order stream + reservoir-sampled fixed-size replay buffer | `--dataset=streaming_<name> --ltm_size=<buffer windows> --n_sample_stm=<live slots> --batch_size=<B>` |
| Streaming (no replay) | in-order stream, most recent frames only | `--dataset=streaming_<name> --batch_size=<B>` (defaults to `--ltm_size=0 --n_sample_stm=<batch size>`) |

With a `streaming_` dataset the model sees a sliding window that advances one frame per step.
`--n_sample_stm` of the `--batch_size` slots hold the current window. The rest are drawn from a replay
buffer of `--ltm_size` window-start indices, maintained by reservoir sampling. Each replayed slot is a
full `--max_frames` window, so the buffer covers `ltm_size × max_frames` frames, which is how the
paper quotes buffer sizes in hours:

| Dataset | `--ltm_size` | K (`--max_frames`) | fps | Buffer |
|---|---|---|---|---|
| Bouncing Balls | 5000 | 10 | 10 | ≈1.4 h |
| 3D Maze | 2500 | 20 | 20 | 42 min |
| Drive | 10000 | 20 | 20 | 2.8 h |
| PLAICraft | 20000 | 10 | 10 | 5.6 h |

Checkpoints are written to `checkpoints/<wandb run id>/`; resume a run with `--resume_id=<wandb run id>`.
The example commands below run on a single GPU; for `N` GPUs prefix a command with `mpiexec -n N`, or `srun --mpi=pmi2 -n N` under SLURM, keeping `--batch_size` divisible by `N`.

**Lifelong Bouncing Balls (`ball_stn`)**

```bash
# VDT-S, Offline Learning
python scripts/video_train_vdt.py --dataset=ball_stn --model_name=VDT-S --max_frames=10 --batch_size=2 --lr=1e-4 --sample_interval=50000 --save_interval=100000
# VDT-S, Lifelong Learning (ER): 5k-window replay buffer (≈1.4 h), 1 of the 2 batch slots from the live stream
python scripts/video_train_vdt.py --dataset=streaming_ball_stn --model_name=VDT-S --max_frames=10 --batch_size=2 --ltm_size=5000 --n_sample_stm=1 --lr=1e-4 --sample_interval=25000 --save_interval=50000
# U-Net, Offline Learning
python scripts/video_train.py --dataset=ball_stn --num_res_blocks=1 --num_channels=64 --max_frames=10 --batch_size=2 --lr=1e-4 --weight_decay=1e-5 --sample_interval=50000 --save_interval=100000
# U-Net, Lifelong Learning (ER)
python scripts/video_train.py --dataset=streaming_ball_stn --num_res_blocks=1 --num_channels=64 --max_frames=10 --batch_size=2 --ltm_size=5000 --n_sample_stm=1 --lr=1e-4 --weight_decay=1e-5 --sample_interval=50000 --save_interval=100000
```

**Lifelong Drive (`drive`)**

```bash
# VDT-M, Offline Learning
python scripts/video_train_vdt.py --dataset=drive --model_name=VDT-M --patch_size=4 --diffusion_space=latent --max_frames=20 --batch_size=8 --lr=1e-4 --weight_decay=1e-6 --sample_interval=50000 --save_interval=100000
# VDT-M, Lifelong Learning (ER): 10k-window replay buffer (≈2.8 h), 2 of the 8 batch slots from the live stream
python scripts/video_train_vdt.py --dataset=streaming_drive --model_name=VDT-M --patch_size=4 --diffusion_space=latent --max_frames=20 --batch_size=8 --ltm_size=10000 --n_sample_stm=2 --lr=1e-4 --weight_decay=1e-6 --clip_grad=1 --sample_interval=50000 --save_interval=100000

# U-Net, Offline Learning
python scripts/video_train.py --dataset=drive --num_res_blocks=1 --num_channels=128 --diffusion_space=latent --max_frames=20 --batch_size=8 --lr=1e-4 --weight_decay=1e-5 --sample_interval=50000 --save_interval=100000
# U-Net, Lifelong Learning (ER)
python scripts/video_train.py --dataset=streaming_drive --num_res_blocks=1 --num_channels=128 --diffusion_space=latent --max_frames=20 --batch_size=8 --ltm_size=10000 --n_sample_stm=2 --lr=1e-4 --weight_decay=1e-5 --sample_interval=50000 --save_interval=100000
```

### Evaluation

Evaluation has two steps: sample continuations from a checkpoint, then score the samples.

```bash
# 1. Autoregressively sample continuations (writes .npy samples + model_config.json)
python scripts/video_sample.py checkpoints/<id>/<ckpt>.pt --T=50 --stop_index=1000 \
    --max_frames=10 --n_obs=5 --sampling_scheme=autoreg --batch_size=50 \
    --sampler=heun-80-inf-0-1-1000-0.002-7-50

# 2. Render sample grids to mp4/gif. Ground truth is shown alongside, so the script needs to know how many videos step 1 generated
python scripts/video_make_mp4.py --eval_dir=<eval_dir> --do_n=8 --num_sampled_videos=1000

# 3. Metrics
python scripts/video_fvd.py    --eval_dir=<eval_dir> --num_videos=1000   # FVD
python scripts/video_jedi.py   --eval_dir=<eval_dir> --num_videos=1000   # JEDi (jedi env)
python scripts/video_loss.py   checkpoints/<id>/<ckpt>.pt --eval_dir=<...> --stop_index=1000 ...  # diffusion loss
python scripts/video_minade.py --eval_dir=<eval_dir> --num_videos=1000   # minADE + ColorKL (balls only)
```

`video_sample.py` writes to
`results/<wandb id>/<ckpt name>_<sampler>/<scheme>_<max_frames>_<T>_<n_obs>_<dataset config>_<lower frame>_<upper frame>_<train|test>/`,
and the metric scripts read their configuration back from that directory name and the saved
`model_config.json`.

Evaluation datasets come in three configurations (`--eval_dataset_config`): `continuous`
(sliding window over every frame), `chunked` (non-overlapping windows), and `default`.

### Configuration

| Variable | Purpose |
|---|---|
| `WANDB_ENTITY`, `WANDB_PROJECT` | Weights & Biases logging destination. If unset, training runs in offline wandb mode. The `--unobserve` flag also forces offline mode. |
| `DATA_ROOT` | Optional. Node-local storage root; dataset files are staged there as they are accessed (useful on clusters). |
| `JEDI_MODEL_DIR` | Optional. Where `videojedi` caches the V-JEPA weights (~10GB, auto-downloaded on first JEDi run). Default: `~/.cache/videojedi`. |

### Repository layout

```
scripts/               Training, sampling, and evaluation entrypoints
improved_diffusion/    Model, diffusion process, datasets, samplers, training loop
datasets/              Download and build scripts; downloaded data lives in datasets/<name>
checkpoints/           Created at runtime; one directory per wandb run id
results/               Created by video_sample.py; metric outputs live next to samples
```

## Citation

```bibtex
@inproceedings{
  yoo2026lifelong,
  title={Lifelong Learning of Video Diffusion Models From a Single Video Stream},
  author={Jason Yoo and Yingchen He and Saeid Naderiparizi and Dylan Green and Gido M. van de Ven and Geoff Pleiss and Frank Wood},
  booktitle={ECCV 2026 Workshop on How to Build Effective World Models for Embodied AI},
  year={2026},
  url={https://openreview.net/forum?id=oW6PbzOHHa}
}
```

## License and acknowledgements

This repository descends from OpenAI's [improved-diffusion](https://github.com/openai/improved-diffusion)
via [flexible-video-diffusion-modeling](https://github.com/plai-group/flexible-video-diffusion-modeling) (MIT).
`improved_diffusion/frechet_video_distance.py` is adapted from
[Google Research](https://github.com/google-research/google-research/tree/master/frechet_video_distance) (Apache 2.0).
`improved_diffusion/vdt.py` is adapted from [VDT](https://github.com/RERV/VDT) (CC BY-NC 4.0), which
derives from Meta's [DiT](https://github.com/facebookresearch/DiT) (CC BY-NC 4.0); that file is
non-commercial use only. The JEDi metric uses [videojedi](https://github.com/oooolga/JEDi) and Meta's
V-JEPA weights. The code is MIT-licensed except as described in `NOTICE`.
