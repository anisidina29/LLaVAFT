import os
import argparse
import json

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor

CKPT = "llava-v1.6-mistral-7b"
SPLIT = "train-00000-of-00738-80a58552f2fb3344-small"
ANSDIR = "./playground/data/eval/websight/answers"
HTMLDIR = "/home/ray/image2code-mar22/data/predictions"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ansdir', type=str, default=ANSDIR)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    parser.add_argument('--split', type=str, default=SPLIT)
    parser.add_argument('--htmldir', type=str, default=HTMLDIR)
    # parser.add_argument('--split', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    curr_dir = os.path.join(args.ansdir, args.split, args.ckpt)
    outdir = f'{args.htmldir}/{args.split}/{args.ckpt}'
    os.makedirs(outdir, exist_ok=True)
    src = os.path.join(curr_dir, 'merge.jsonl')
    data = []
    with open(src, 'r') as f:
        for line in f:
            if not line:
                break
            record = json.loads(line)
            data.append(record)
            code = record['text']
            html_fname = f'{outdir}/{record["question_id"]}.html'
            with open(html_fname, 'w') as htmlf:
                htmlf.write(code)
    