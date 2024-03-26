import argparse
import os

import clip
import numpy as np
import torch

from bs4 import BeautifulSoup
from datetime import datetime
from difflib import SequenceMatcher
from PIL import Image
from tqdm import tqdm


def rescale_and_mask(image_path):
    with Image.open(image_path) as img:
        width, height = img.size

        # Determine which side is shorter
        if width < height:
            # Width is shorter, scale height to match the width
            new_size = (width, width)
        else:
            # Height is shorter, scale width to match the height
            new_size = (height, height)

        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized


def get_clip_feats(model, preprocess, image_path, device):
    img_resized = preprocess(rescale_and_mask(image_path)).unsqueeze(0).to(device)
    return model.encode_image(img_resized)


def get_clip_similarity(model, preprocess, pred_path, ref_path, device):
    pred_feats = get_clip_feats(model, preprocess, pred_path.replace('.html', '.png'), device)
    ref_feats = get_clip_feats(model, preprocess, ref_path.replace('.html', '.png'), device)
    # Normalize features
    pred_feats /= pred_feats.norm(dim=-1, keepdim=True)
    ref_feats /= ref_feats.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (pred_feats @ ref_feats.T).item()

    return similarity

def extract_text(html_url):
    with open(html_url, 'r') as f:
        html_content = f.read().strip()
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract all text from the parsed HTML
    text = soup.get_text()
    # Print the extracted text
    return ' '.join(text.split())


def get_text_similarity(pred_path, ref_path):
    pred_text = extract_text(pred_path)
    ref_text = extract_text(ref_path)
    return SequenceMatcher(None, pred_text, ref_text).ratio()


def evaluate_pair(clip_model, clip_preprocess, pred_path, ref_path, device):
    return {
            'text_sim': get_text_similarity(pred_path, ref_path),
            'clip_sim': get_clip_similarity(clip_model, clip_preprocess, pred_path, ref_path, device)
        }


def eval_batch(pred_dir, ref_dir, clip_model_name='ViT-B/32'):
    pred_fnames = [] # list of generated html files to evaluate
    ref_fnames = [] # list of ref html files

    ## check if all the prediction files have ground truths
    for filename in sorted(os.listdir(pred_dir)):
        if filename.endswith(".html"):
            if os.path.exists(os.path.join(ref_dir, filename)):
                pred_fnames.append(os.path.join(pred_dir, filename))
                ref_fnames.append(os.path.join(ref_dir, filename))

    assert len(pred_fnames) == len(ref_fnames)
    print ("total #egs: ", len(pred_fnames))
    text_sims, clip_sims = {}, {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    for i in tqdm(range(len(ref_fnames))):
        ref_path = ref_fnames[i]
        fname = ref_path.split('/')[-1][:-5]
        pred_path = pred_fnames[i]
        if not fname in pred_path:
            print(f'ref_path is {fname} but pred_path is {pred_path}')
            continue
        scores = evaluate_pair(clip_model, clip_preprocess, pred_path, ref_path, device)
        text_sims[fname] = scores['text_sim']
        clip_sims[fname] = scores['clip_sim']
    return text_sims, clip_sims


def get_score_stats(sims, outfile, corrupt_threshold=0.1, bad_threshold=0.5):
    n = len(sims)
    a = np.asarray(list(sims.values()))
    names = np.asarray(list(sims.keys()))
    corrupts = names[a < corrupt_threshold]
    nc = len(corrupts)
    outfile.write(f'{nc} samples ({nc * 100 / n:.2f}%) with sim score < {corrupt_threshold}\n')
    for name in corrupts:
        outfile.write(f'\t{name}: {sims[name]}\n')
    bads = names[a < bad_threshold]
    nb = len(bads)
    outfile.write(f'{nb} samples ({nb * 100 / n:.2f}%) with sim score < {bad_threshold}\n')
    for name in bads:
        outfile.write(f'\t{name}: {sims[name]}\n')
    sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
    best = sorted_sims[0]
    worst = sorted_sims[-1]
    outfile.write(f'Mean: {a.mean():.6f} | Median: {np.median(a):.6f}\n')
    outfile.write(f'Max similarity: {best[-1]:.6f} by {best[0]}\n')
    outfile.write(f'Min similarity: {worst[-1]:.6f} by {worst[0]}\n')


def report_stats(text_sims, clip_sims, outfile):    
    outfile.write('TEXT SIMILARITY\n')
    get_score_stats(text_sims, outfile, corrupt_threshold=0.1, bad_threshold=0.6)
    outfile.write('=' * 30 + '\n')
    outfile.write('CLIP SIMILARITY\n')
    get_score_stats(clip_sims, outfile, corrupt_threshold=0.1, bad_threshold=0.6)


RESDIR = '/home/ray/image2code-mar22/results'
REFDIR = '/efs/shared_storage/img2code/eval-d2c'
HTMLDIR ='/efs/shared_storage/img2code/predictions'
SPLIT = 'train-00000-of-00738-80a58552f2fb3344-small'
CKPT = 'llava-v1.6-mistral-7b'
CLIP_MODEL_NAME = 'ViT-B/32'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resdir', type=str, default=RESDIR)
    parser.add_argument('--refdir', type=str, default=REFDIR)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    parser.add_argument('--split', type=str, default=SPLIT)
    parser.add_argument('--htmldir', type=str, default=HTMLDIR)
    parser.add_argument('--clip', type=str, default=CLIP_MODEL_NAME)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred_dir = os.path.join(args.htmldir, args.split, args.ckpt)
    text_sims, clip_sims = eval_batch(pred_dir, args.refdir)
    stats_dir = os.path.join(args.resdir, args.split, args.ckpt)
    os.makedirs(stats_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    fname = f'{stats_dir}/{ts}.txt'
    outfile = open(fname, 'w')
    outfile.write(f'Partition: {args.split}.\n{len(text_sims)} samples.\n')
    outfile.write(f'Model: {args.ckpt}\n')

    report_stats(text_sims, clip_sims, outfile)
    outfile.close()
    