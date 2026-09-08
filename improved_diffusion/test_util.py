import argparse
import os
from filelock import FileLock
from pathlib import Path
import torch as th
import numpy as np
from PIL import Image
import imageio


class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)


def get_model_results_path(args):
    """
        Given arguments passed to an evaluation run, returns the path to the results path.
        The path has the format "results/<checkpoint_dir_subpath>/checkpoint name" where
        <checkpoint_dir_subpath> is the subset of checkpoint path after ".*checkpoint.*/"
        For example, if "checkpoints/abcdefg/ema_latest.pt"
        is the checkpoint path, the result path will be
        "results/abcdefg/ema_latest_<checkpoint_step>/". In this path, <checkpoint_step> is the
        training step of the checkpoint and will only be added if the checkpoint path ends with
        "latest", since otherwise the checkpoint name itself ends with the step number.
        If args.eval_dir is not None, this function does nothing and returns the same path.
        args is expected to have the following attributes:
        - checkpoint_path
        - eval_dir
        - sampler
        - timestep_respacing
    """
    # Extract the diffusion sampling arguments string (DDIM/respacing)
    postfix = ""
    postfix += f"_{args.sampler}"
    if args.timestep_respacing != "":
        postfix += "_" + f"respace{args.timestep_respacing}"

    # Create the output directory (if does not exist)
    if args.eval_dir is None:
        checkpoint_path = Path(args.checkpoint_path)
        name = f"{checkpoint_path.stem}"
        if name.endswith("latest"):
            checkpoint_step = th.load(args.checkpoint_path, map_location="cpu")["step"]
            name += f"_{checkpoint_step}"
        if postfix != "":
            name += postfix
        path = None
        for idx, x in enumerate(checkpoint_path.parts):
            if "checkpoint" in x:
                path = Path(*(checkpoint_path.parts[idx+1:]))
                break
        assert path is not None
        return Path("results") / path.parent / name
    else:
        return Path(args.eval_dir)


def get_eval_run_identifier(args):
    res = args.sampling_scheme
    res += f"_{args.max_frames}_{args.T}_{args.n_obs}_{args.eval_dataset_config}"
    res += f"_{args.lower_frame_range}_{args.upper_frame_range}"
    res += f"_train" if args.eval_on_train else "_test"
    return res


def parse_eval_run_identifier(identifier):
    split = identifier.split("_")
    return dict(
        sampling_scheme=split[0], max_frames=int(split[1]), T=int(split[2]), n_obs=int(split[3]),
        eval_dataset_config=split[4], lower_frame_range=int(split[5]),
        upper_frame_range=None if split[6]=="None" else int(split[6]),
        eval_on_train=split[7] == "train"
    )


################################################################################
#                               Model loading                                  #
################################################################################
def model_args_from_config(config, **overrides):
    """
    Turn a saved training config (the checkpoint's "config" entry, or model_config.json) into the
    Namespace that create_model_and_diffusion expects at evaluation time: decoding is enabled and
    `overrides` (e.g. timestep_respacing) replace the training-time values. Checkpoints written
    before `model_type` was recorded are patched to the values those runs used.
    """
    config = {**config, **overrides}
    if "diffusion_space_kwargs" in config:  # absent from configs written before latent-space support existed
        config["diffusion_space_kwargs"] = {**config["diffusion_space_kwargs"], "enable_decoding": True}
    model_args = argparse.Namespace(**config)
    if not hasattr(model_args, "model_type"):
        is_vdt = hasattr(model_args, "model_name")
        model_args.model_type = "vdt" if is_vdt else "unet"
        if is_vdt and not hasattr(model_args, "input_size"):
            model_args.input_size = model_args.image_size
        if is_vdt and not hasattr(model_args, "patch_size"):
            model_args.patch_size = 2
    return model_args


def load_model_from_checkpoint(checkpoint_path, device, **overrides):
    """Load a training checkpoint and build its model and diffusion in eval mode. See model_args_from_config."""
    # Imported here so that scripts which never load a checkpoint (FVD, JEDi, minADE) do not load the model code.
    from improved_diffusion import dist_util
    from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
    data = dist_util.load_state_dict(checkpoint_path, map_location="cpu")
    model_args = model_args_from_config(data["config"], **overrides)
    model, diffusion = create_model_and_diffusion(
        model_type=model_args.model_type,
        **args_to_dict(model_args, model_and_diffusion_defaults(model_type=model_args.model_type).keys()),
    )
    model.load_state_dict(data["state_dict"])
    model = model.to(device)
    model.eval()
    return model, diffusion, model_args


################################################################################
#                            Evaluation datasets                               #
################################################################################
class SampleDataset(th.utils.data.Dataset):
    """Videos written by scripts/video_sample.py (uint8 .npy files, TxCxHxW), returned scaled to [-1, 1]."""
    def __init__(self, samples_path, sample_idx, length, start_idx=0):
        self.samples_path = Path(samples_path)
        self.start_idx = start_idx
        self.sample_idx = sample_idx
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.samples_path / f"sample_{self.start_idx+idx:04d}-{self.sample_idx}.npy"
        npy = np.load(path).astype(np.float32)
        normed = -1 + 2 * npy / 255
        return th.tensor(normed).type(th.float32), {}


class DecodedDataset(th.utils.data.Dataset):
    """
    Decodes a latent-space dataset to pixels with the Stable Diffusion VAE, caching every item as a
    .npy file under cache_path. Items are locked while being written so that concurrent evaluation
    jobs share the decoding work instead of repeating it.
    """
    def __init__(self, encoded_dataset, cache_path, decode_chunk_size,
                 pre_decode=False, subset_indices=None):
        self.encoded_dataset = encoded_dataset
        self.cache_path = Path(cache_path)
        self.decode_chunk_size = decode_chunk_size
        self.vae = None
        if pre_decode:
            self.pre_decode(subset_indices)

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        path = self.cache_path / f"sample_{idx:04d}.npy"
        with Protect(path, timeout=3600):
            if not path.exists():
                print(f"Decoding data item {idx}...")
                encoding, _ = self.encoded_dataset[idx]
                video = self._decode(encoding)
                np.save(path, video)
                print(f"Finished decoding data item {idx}.")
        npy = np.load(path).astype(np.float32)
        normed = -1 + 2 * npy / 255
        return th.tensor(normed).type(th.float32), {}

    @th.no_grad()
    def _decode(self, encoding):
        no_vae = self.vae is None
        if no_vae:
            self._initialize_vae()
        with th.no_grad():
            decoded = [self.vae.decode(encoding[j:j+self.decode_chunk_size].to(self.vae.device)/0.13025
                       ).sample for j in range(0, encoding.shape[0], self.decode_chunk_size)]
        drange = [-1, 1]
        decoded = th.cat(decoded, dim=0).cpu().clamp(*drange).numpy()
        decoded = (decoded - drange[0]) / (drange[1] - drange[0]) * 255
        if no_vae:
            self._remove_vae()
        return decoded.astype(np.uint8)

    def pre_decode(self, subset_indices=None):
        self._initialize_vae()
        init_indices = [i for i in range(len(self))] if subset_indices is None else subset_indices
        for i in np.random.permutation(init_indices):  # random ordering
            self[i]
        self._remove_vae()

    def _initialize_vae(self):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=th.float16)
        self.vae.eval()
        if th.cuda.is_available():
            self.vae = self.vae.cuda()

    def _remove_vae(self):
        del self.vae
        self.vae = None
        import gc
        gc.collect()
        th.cuda.empty_cache()


def decoded_ground_truth_cache_dir(root, dataset_name, T, eval_dataset_config, frame_range, eval_on_train):
    """Cache directory for DecodedDataset, keyed on everything that determines which ground-truth clips are used."""
    dataset_name = dataset_name.removeprefix("streaming_")
    path = Path(root) / f"{dataset_name}_{T}_{eval_dataset_config}_{frame_range[0]}_{frame_range[1]}_{eval_on_train}"
    path.mkdir(parents=True, exist_ok=True)
    return path


################################################################################
#                           Visualization functions                            #
################################################################################
def mark_as_observed(images, color=[255, 0, 0]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


def tensor2pil(tensor, drange=[0,1]):
    """Given a tensor of shape (Bx)3xwxh with pixel values in drange, returns a PIL image
       of the tensor. Returns a list of images if the input tensor is a batch.
    Args:
        tensor: A tensor of shape (Bx)3xwxh
        drange (list, optional): Range of pixel values in the input tensor. Defaults to [0,1].
    """
    assert tensor.ndim == 3 or tensor.ndim == 4
    if tensor.ndim == 3:
        return tensor2pil(tensor.unsqueeze(0), drange=drange)[0]
    img_batch = tensor.cpu().numpy().transpose([0, 2, 3, 1])
    img_batch = (img_batch - drange[0]) / (drange[1] - drange[0])  * 255 # img_batch with pixel values in [0, 255]
    img_batch = img_batch.astype(np.uint8)
    return [Image.fromarray(img) for img in img_batch]

def tensor2gif(tensor, path, drange=[0, 1], random_str=""):
    frames = tensor2pil(tensor, drange=drange)
    tmp_path = f"/tmp/tmp_{random_str}.png"
    res = []
    for frame in frames:
        frame.save(tmp_path)
        res.append(imageio.imread(tmp_path))
    imageio.mimsave(path, res)

def tensor2mp4(tensor, path, drange=[0, 1], random_str=""):
    gif_path = f"/tmp/tmp_{random_str}.gif"
    tensor2gif(tensor, path=gif_path, drange=drange, random_str=random_str)
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i {gif_path} -r 10 -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" {path}")