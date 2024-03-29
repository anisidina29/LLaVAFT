import argparse
import json
import os
import shutil

CKPT = "llava-v1.6-mistral-7b"
SPLIT = "train-00000-of-00738-80a58552f2fb3344-small"
ANSDIR = "./playground/data/eval/websight/answers"
HTMLDIR = "/home/ray/image2code-mar22/data/predictions"


def clean_html(text):
    # Locate the start of <html> tag
    start = text.find('<html')
    if start == -1:
        # the html is likely corrupted
        return text
    text = text[start:]
    # Locate the end of </html> tag
    closing_idx = text.find('</html>')
    if closing_idx == -1:
        return text
    return text[:closing_idx + len('</html>')] + '\n'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ansdir', type=str, default=ANSDIR)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    parser.add_argument('--split', type=str, default=SPLIT)
    parser.add_argument('--htmldir', type=str, default=HTMLDIR)
    parser.add_argument('--ts', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    curr_dir = os.path.join(args.ansdir, args.ckpt, args.split, args.ts)
    outdir = f'{args.htmldir}/{args.ckpt}/{args.split}/{args.ts}'
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(os.path.join(curr_dir, 'config.json'), os.path.join(outdir, 'config.json'))
    src = os.path.join(curr_dir, 'merge.jsonl')
    data = []
    with open(src, 'r') as f:
        for line in f:
            if not line:
                break
            record = json.loads(line)
            data.append(record)
            code = record['text']
            clean_code = clean_html(code)
            html_fname = f'{outdir}/{record["question_id"]}.html'
            with open(html_fname, 'w') as htmlf:
                htmlf.write(clean_code)
    