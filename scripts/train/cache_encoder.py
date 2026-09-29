import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from scripts.train.dual_wm import get_img_preprocessor
from stable_worldmodel.wm.utils import load_pretrained


def encoder_fingerprint(model):
    digest = hashlib.blake2b()
    for prefix in ['encoder', 'projector']:
        for key, value in sorted(getattr(model, prefix).state_dict().items()):
            digest.update((prefix + '.' + key).encode())
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build(args):
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(target)
    source = Path(args.dataset).resolve()
    checkpoint = Path(args.checkpoint).expanduser()
    model = load_pretrained(checkpoint.resolve() if checkpoint.exists() else args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    signature = encoder_fingerprint(model)
    transform = get_img_preprocessor('pixels', 'pixels', args.img_size)
    with h5py.File(source, 'r') as data:
        total = len(data['pixels'])
        stop = total if args.stop is None else args.stop
        if not 0 <= args.start < stop <= total:
            raise ValueError('Invalid cache row range')
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, 'x') as output:
            output.attrs.update(source_dataset=str(source), encoder_signature=signature, img_size=args.img_size, row_start=args.start, row_stop=stop, source_rows=total, complete=False, autocast_dtype=args.precision)
            emb = None
            with torch.inference_mode():
                for begin in range(args.start, stop, args.batch_size):
                    end = min(begin + args.batch_size, stop)
                    pixels = torch.from_numpy(data['pixels'][begin:end]).permute(0, 3, 1, 2)
                    pixels = transform({'pixels': pixels})['pixels'].cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                        features = model.encode({'pixels': pixels.unsqueeze(0)})['emb'][0]
                    if emb is None:
                        output.attrs['embedding_dtype'] = str(features.dtype)
                        emb = output.create_dataset('emb', shape=(stop - args.start, features.shape[-1]), dtype='float32', chunks=(min(1024, stop-args.start), features.shape[-1]), compression=hdf5plugin.Blosc(cname='lz4', clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE))
                    emb[begin-args.start:end-args.start] = features.float().cpu().numpy()
                    if (begin - args.start) // args.batch_size % 100 == 0:
                        output.flush()
                        print(json.dumps(dict(rows_done=end-args.start, rows_total=stop-args.start)), flush=True)
            output.attrs['complete'] = True
    print(json.dumps(dict(completed=str(target), rows=stop-args.start)), flush=True)


def merge(args):
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(target)
    parts = []
    for path in args.parts:
        path = Path(path).resolve()
        with h5py.File(path, 'r') as data:
            if not data.attrs['complete']:
                raise ValueError(f'Incomplete cache: {path}')
            parts.append((int(data.attrs['row_start']), int(data.attrs['row_stop']), path, dict(data.attrs), data['emb'].shape))
    parts.sort()
    first = parts[0][3]
    end = 0
    for start, stop, path, attrs, shape in parts:
        if start != end:
            raise ValueError('Cache parts must cover every row without overlap')
        if len(shape) != 2 or shape[0] != stop - start or shape[1] != parts[0][4][-1] or stop <= start:
            raise ValueError('Cache part shape does not match its row range and embedding dimension')
        for key in ['source_dataset', 'encoder_signature', 'img_size', 'source_rows', 'embedding_dtype', 'autocast_dtype']:
            if attrs[key] != first[key]:
                raise ValueError(f'Cache metadata differs: {key}')
        end = stop
    if end != first['source_rows']:
        raise ValueError('Cache does not cover the full dataset')
    layout = h5py.VirtualLayout(shape=(end, parts[0][4][-1]), dtype='float32')
    for start, stop, path, attrs, shape in parts:
        layout[start:stop] = h5py.VirtualSource(str(path), 'emb', shape=shape)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, 'x', libver='latest') as output:
        output.create_virtual_dataset('emb', layout)
        with h5py.File(first['source_dataset'], 'r') as source:
            for key in source:
                if key != 'pixels':
                    output[key] = h5py.ExternalLink(first['source_dataset'], '/' + key)
        output.attrs.update(first)
        output.attrs.update(row_start=0, row_stop=end, complete=True, cache_parts=json.dumps([str(p[2]) for p in parts]))
    print(json.dumps(dict(merged=str(target), rows=end)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--dataset', required=True)
    build_parser.add_argument('--checkpoint', required=True)
    build_parser.add_argument('--output', required=True)
    build_parser.add_argument('--start', type=int, default=0)
    build_parser.add_argument('--stop', type=int)
    build_parser.add_argument('--batch-size', type=int, default=128)
    build_parser.add_argument('--img-size', type=int, default=224)
    build_parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    merge_parser = sub.add_parser('merge')
    merge_parser.add_argument('--output', required=True)
    merge_parser.add_argument('--parts', nargs='+', required=True)
    args = parser.parse_args()
    (build if args.command == 'build' else merge)(args)


if __name__ == '__main__':
    main()
