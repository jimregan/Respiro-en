#!/usr/bin/env python3
"""Train Respiro-en from CSV breath interval annotations."""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FRAME_HOP_SECONDS = 0.01


@dataclass(frozen=True)
class Example:
    path: str
    breaths: tuple


def read_manifest(path):
    examples = []
    with open(path, newline='', encoding='utf-8') as manifest:
        reader = csv.DictReader(manifest)
        if reader.fieldnames is None or not {'path', 'breath'} <= set(reader.fieldnames):
            raise ValueError(f'{path} must contain path and breath columns')
        for line_number, row in enumerate(reader, start=2):
            try:
                raw_breaths = json.loads(row['breath'])
                breaths = tuple((float(start), float(end)) for start, end in raw_breaths)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'invalid breath annotation at {path}:{line_number}') from error
            for start, end in breaths:
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(
                        f'invalid interval [{start}, {end}] at {path}:{line_number}'
                    )
            examples.append(Example(row['path'], breaths))
    if not examples:
        raise ValueError(f'{path} contains no examples')
    return examples


def intervals_to_frames(intervals, frame_count, hop_seconds=FRAME_HOP_SECONDS):
    """Rasterize half-open time intervals into a frame-wise float target."""
    target = torch.zeros(frame_count, dtype=torch.float32)
    for start, end in intervals:
        first = max(0, math.floor(start / hop_seconds))
        last = min(frame_count, math.ceil(end / hop_seconds))
        if first < last:
            target[first:last] = 1.0
    return target


def resolve_audio_path(audio_root, manifest_path, extension='.wav'):
    relative = Path(manifest_path)
    candidate = audio_root / relative
    if candidate.is_file():
        return candidate
    if not relative.suffix:
        candidate = candidate.with_suffix(extension)
        if candidate.is_file():
            return candidate
        parts = relative.name.split('_')
        if len(parts) >= 3:
            candidate = audio_root / parts[0] / parts[1] / relative.name
            candidate = candidate.with_suffix(extension)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f'audio not found for manifest path {manifest_path!r}')


class BreathDataset(Dataset):
    def __init__(self, manifest, audio_root, sample_rate=16000, extension='.wav'):
        self.examples = read_manifest(manifest)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.extension = extension

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        try:
            import librosa
        except ImportError as error:
            raise RuntimeError('training requires librosa>=0.10.0') from error
        from modules import feature_extractor

        example = self.examples[index]
        audio_path = resolve_audio_path(self.audio_root, example.path, self.extension)
        wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        feature, length = feature_extractor(wav, sr=self.sample_rate)
        feature = feature.squeeze(0)
        frame_count = int(length.item())
        target = intervals_to_frames(example.breaths, frame_count)
        return feature, target, frame_count


def collate_examples(batch):
    features, targets, lengths = zip(*batch)
    max_length = max(lengths)
    feature_batch = features[0].new_zeros(
        (len(batch), features[0].shape[0], features[0].shape[1], max_length)
    )
    target_batch = targets[0].new_zeros((len(batch), max_length))
    for index, (feature, target, length) in enumerate(batch):
        feature_batch[index, :, :, :length] = feature[:, :, :length]
        target_batch[index, :length] = target[:length]
    return feature_batch, target_batch, torch.tensor(lengths, dtype=torch.long)


def valid_frame_mask(lengths, max_length):
    frames = torch.arange(max_length, device=lengths.device)
    return frames.unsqueeze(0) < lengths.unsqueeze(1)


def masked_weighted_bce(probabilities, targets, lengths, positive_weight):
    mask = valid_frame_mask(lengths, probabilities.shape[1])
    weights = torch.where(targets > 0.5, positive_weight, 1.0)
    losses = F.binary_cross_entropy(probabilities, targets, reduction='none')
    return (losses * weights * mask).sum() / mask.sum().clamp_min(1)


def update_counts(counts, probabilities, targets, lengths, threshold):
    mask = valid_frame_mask(lengths, probabilities.shape[1])
    predictions = probabilities >= threshold
    positives = targets >= 0.5
    counts['tp'] += int((predictions & positives & mask).sum().item())
    counts['fp'] += int((predictions & ~positives & mask).sum().item())
    counts['fn'] += int((~predictions & positives & mask).sum().item())


def counts_to_metrics(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def run_epoch(model, loader, device, positive_weight, threshold, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    batches = 0
    counts = {'tp': 0, 'fp': 0, 'fn': 0}

    for features, targets, lengths in loader:
        features = features.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        with torch.set_grad_enabled(training):
            probabilities = model(features, lengths)
            loss = masked_weighted_bce(
                probabilities, targets, lengths, positive_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        loss_total += float(loss.item())
        batches += 1
        update_counts(counts, probabilities.detach(), targets, lengths, threshold)

    metrics = counts_to_metrics(counts)
    metrics['loss'] = loss_total / max(batches, 1)
    return metrics


def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path, model, optimizer, epoch, metrics, arguments):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'args': vars(arguments),
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--dev-csv', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path)
    parser.add_argument('--train-audio-root', type=Path)
    parser.add_argument('--dev-audio-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--audio-extension', default='.wav')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--positive-weight', type=float, default=5.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or mps')
    parser.add_argument('--init-checkpoint', type=Path)
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        raise SystemExit('epochs and batch size must be positive')
    if arguments.positive_weight <= 0:
        raise SystemExit('positive weight must be positive')
    train_audio_root = arguments.train_audio_root or arguments.audio_root
    dev_audio_root = arguments.dev_audio_root or arguments.audio_root
    if train_audio_root is None or dev_audio_root is None:
        raise SystemExit(
            'provide --audio-root or both --train-audio-root and --dev-audio-root'
        )

    seed_everything(arguments.seed)
    device = choose_device(arguments.device)
    print(f'Using device: {device}')

    from modules import DetectionNet

    train_dataset = BreathDataset(
        arguments.train_csv, train_audio_root, extension=arguments.audio_extension
    )
    dev_dataset = BreathDataset(
        arguments.dev_csv, dev_audio_root, extension=arguments.audio_extension
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        collate_fn=collate_examples,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        collate_fn=collate_examples,
    )

    model = DetectionNet().to(device)
    if arguments.init_checkpoint:
        checkpoint = torch.load(arguments.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'Initialized from {arguments.init_checkpoint}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)

    best_f1 = -1.0
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, arguments.positive_weight,
            arguments.threshold, optimizer
        )
        with torch.no_grad():
            dev_metrics = run_epoch(
                model, dev_loader, device, arguments.positive_weight,
                arguments.threshold
            )
        print(
            f'Epoch {epoch:03d} '
            f'train loss={train_metrics["loss"]:.4f} f1={train_metrics["f1"]:.4f} '
            f'dev loss={dev_metrics["loss"]:.4f} '
            f'precision={dev_metrics["precision"]:.4f} '
            f'recall={dev_metrics["recall"]:.4f} f1={dev_metrics["f1"]:.4f}'
        )
        save_checkpoint(
            arguments.output_dir / 'latest.pt', model, optimizer, epoch,
            dev_metrics, arguments
        )
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            save_checkpoint(
                arguments.output_dir / 'best.pt', model, optimizer, epoch,
                dev_metrics, arguments
            )


if __name__ == '__main__':
    main()
