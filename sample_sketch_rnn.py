import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from sketch_rnn import make_image
from sketch_rnn.dataset import SketchRNNDataset, collate_drawings, load_strokes
from sketch_rnn.hparams import hparam_parser
from sketch_rnn.model import (
    SketchRNN,
    model_step,
    sample_conditional,
    sample_unconditional,
)
from sketch_rnn.utils import AverageMeter, ModelCheckpoint


def sample_sketch_rnn(args):
    torch.manual_seed(884)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    model = SketchRNN(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pt")))

    T = 0.1
    sample_dir = os.path.join(args.save_dir, "unconditional_samples")
    os.makedirs(sample_dir, exist_ok=True)

    for i in trange(100):
        x_samp, z_samp = sample_unconditional(model, T=T, device=device)
        sequence = torch.cat([x_samp, z_samp.unsqueeze(1)], dim=1).cpu()
        make_image(sequence, i, sample_dir)

    train_strokes, valid_strokes, test_strokes = load_strokes(args.data_dir, args)
    train_data = SketchRNNDataset(
        train_strokes,
        max_len=args.max_seq_len,
        random_scale_factor=args.random_scale_factor,
        augment_stroke_prob=args.augment_stroke_prob,
    )
    val_data = SketchRNNDataset(
        valid_strokes,
        max_len=args.max_seq_len,
        scale_factor=train_data.scale_factor,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0,
    )

    # initialize data loaders
    collate_fn = lambda x: collate_drawings(x, args.max_seq_len)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=use_gpu,
        num_workers=args.num_workers,
    )

    cond_sample_dir = os.path.join(args.save_dir, "conditional_samples")
    os.makedirs(cond_sample_dir, exist_ok=True)

    for idx, (data, lengths) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        print(data.shape, lengths.shape)
        x_samp, z_samp = sample_conditional(model, data, lengths, T=T, device=device)
        sequence = torch.cat([x_samp, z_samp.unsqueeze(1)], dim=1).cpu()
        make_image(sequence, idx, cond_sample_dir)


if __name__ == "__main__":
    hp_parser = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp_parser])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    sample_sketch_rnn(args)
