"""
Train a U-Net video diffusion model.

The CLI and training entry point are shared with scripts/video_train_vdt.py and live in
improved_diffusion/train_entry.py (the loop itself is improved_diffusion.train_util.TrainLoop).
"""

from improved_diffusion.train_entry import main


if __name__ == "__main__":
    main(model_type="unet")
