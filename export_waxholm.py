#!/usr/bin/env python3
"""Export Waxholm TextGrid breath annotations as Respiro CSV manifests."""

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_BREATH_TAGS = ('XinandX', 'XutandX')
ITEM_RE = re.compile(r'item\s*\[\d+\]:')
NAME_RE = re.compile(r'name\s*=\s*"([^"]*)"')
INTERVAL_RE = re.compile(
    r'intervals\s*\[\d+\]:\s*'
    r'xmin\s*=\s*([0-9.eE+\-]+)\s*'
    r'xmax\s*=\s*([0-9.eE+\-]+)\s*'
    r'text\s*=\s*"([^"]*)"'
)


def parse_tier(path, tier_name):
    text = path.read_text(encoding='utf-8', errors='replace')
    for block in ITEM_RE.split(text)[1:]:
        name = NAME_RE.search(block)
        if name and name.group(1) == tier_name:
            return [
                (float(start), float(end), label)
                for start, end, label in INTERVAL_RE.findall(block)
            ]
    return []


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def speaker_from_stem(stem):
    return stem.split('.', 1)[0]


def collect_examples(textgrid_root, audio_root, breath_tags):
    examples = []
    missing_audio = 0
    missing_words_tier = 0
    tag_counts = Counter()
    for textgrid_path in sorted(textgrid_root.rglob('*.textgrid')):
        relative = textgrid_path.relative_to(textgrid_root).with_suffix('.wav')
        audio_path = audio_root / relative
        if not audio_path.is_file():
            missing_audio += 1
            continue
        words = parse_tier(textgrid_path, 'words')
        if not words:
            missing_words_tier += 1
            continue
        intervals = []
        for start, end, label in words:
            if label in breath_tags:
                intervals.append((start, end))
                tag_counts[label] += 1
        examples.append({
            'path': relative.as_posix(),
            'breath': merge_intervals(intervals),
            'speaker': speaker_from_stem(textgrid_path.stem),
        })
    return examples, tag_counts, missing_audio, missing_words_tier


def split_speakers(examples, dev_fraction, test_fraction, seed):
    speakers = sorted({example['speaker'] for example in examples})
    random.Random(seed).shuffle(speakers)
    dev_count = round(len(speakers) * dev_fraction)
    test_count = round(len(speakers) * test_fraction)
    dev_speakers = set(speakers[:dev_count])
    test_speakers = set(speakers[dev_count:dev_count + test_count])

    splits = {'train': [], 'dev': [], 'test': []}
    for example in examples:
        if example['speaker'] in dev_speakers:
            split = 'dev'
        elif example['speaker'] in test_speakers:
            split = 'test'
        else:
            split = 'train'
        splits[split].append(example)
    return splits


def write_manifest(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['path', 'breath'])
        writer.writeheader()
        for example in examples:
            writer.writerow({
                'path': example['path'],
                'breath': json.dumps(example['breath'], ensure_ascii=False),
            })


def copy_audio_files(examples, audio_root, destination):
    copied = 0
    for example in examples:
        relative = Path(example['path'])
        source = audio_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--textgrid-root', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('datasets/waxholm'))
    parser.add_argument(
        '--copy-audio', action='store_true',
        help='copy referenced WAVs to OUTPUT_DIR/audio for a portable dataset'
    )
    parser.add_argument(
        '--breath-tag', action='append', dest='breath_tags',
        help='word-tier tag to treat as breath; repeatable (default: XinandX, XutandX)'
    )
    parser.add_argument('--dev-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=15)
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.dev_fraction < 0 or arguments.test_fraction < 0:
        raise SystemExit('split fractions cannot be negative')
    if arguments.dev_fraction + arguments.test_fraction >= 1:
        raise SystemExit('dev and test fractions must sum to less than 1')
    breath_tags = set(arguments.breath_tags or DEFAULT_BREATH_TAGS)
    examples, tag_counts, missing_audio, missing_words_tier = collect_examples(
        arguments.textgrid_root, arguments.audio_root, breath_tags
    )
    if not examples:
        raise SystemExit('no matching TextGrid and WAV pairs found')

    splits = split_speakers(
        examples, arguments.dev_fraction, arguments.test_fraction, arguments.seed
    )
    for split, split_examples in splits.items():
        path = arguments.output_dir / f'{split}.csv'
        write_manifest(path, split_examples)
        positive = sum(bool(example['breath']) for example in split_examples)
        speakers = len({example['speaker'] for example in split_examples})
        print(
            f'{split}: {len(split_examples)} utterances, {positive} with breaths, '
            f'{speakers} speakers -> {path}'
        )
    if arguments.copy_audio:
        copied = copy_audio_files(
            examples, arguments.audio_root, arguments.output_dir / 'audio'
        )
        print(f'copied {copied} WAV files -> {arguments.output_dir / "audio"}')
    print(f'breath intervals by tag: {dict(sorted(tag_counts.items()))}')
    print(f'skipped: {missing_audio} without audio, {missing_words_tier} without words tier')


if __name__ == '__main__':
    main()
