import os
import cv2
import glob
import math
import torch
import datasets
import scipy.io
import jsonlines
import numpy as np

from tqdm import tqdm
from PIL import Image
from clip_interrogator import Config, Interrogator
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

MAIN_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

# img_dir = f'{MAIN_DIR}/data/danbooru-2020-512'
img_dir = f'{MAIN_DIR}/data/waymo/images'

jlname = f"{MAIN_DIR}/data/waymo-200.jsonl"
# def make_prompt():
#     # imgs = sorted(glob.glob(f'{MAIN_DIR}/data/waymo/**/*.jpg', recursive=True) + \
#                 #   glob.glob(f'{MAIN_DIR}/data/waymo/**/*.png', recursive=True))[:20000]
#     imgs = sorted(glob.glob(f'{MAIN_DIR}/data/waymo/**/*.jpg', recursive=True) + \
#                   glob.glob(f'{MAIN_DIR}/data/waymo/**/*.png', recursive=True))[:20000]
#     if not os.path.exists(jlname + '.done'):
#         ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
#         with jsonlines.open(jlname, 'w', flush=True) as w:
#             for img_path in tqdm(imgs):
#                 with torch.no_grad():
#                     image = Image.open(img_path).convert('RGB')
#                     prompt = ci.interrogate_fast(image)
#                     w.write({
#                         "image": img_path, 
#                         "text": prompt}
#                     )
#         with open(jlname + '.done', 'w', encoding='utf-8') as f:
#             f.write('Done.')



# for name in ['erika', 'illyasviel', 'infor', 'muko']:
#     with jsonlines.open(jlname, 'r') as r:
#         with jsonlines.open(jlname.replace('/data/waymo-200', f'/data/waymo-200-{name}'), 'w', flush=True) as w:
#             for item in r.iter():
#                 w.write({
#                     "image": item["image"].replace('/data/waymo-200', f'/data/waymo-200-{name}')
#                 })

def save_to_disk():
    files = glob.glob(
        os.path.join(
            MAIN_DIR, 
            'data/waymo', 
            '**', '*.jpg'), 
        recursive=True)
    files = sorted(files)
    json_file = os.path.join(
        MAIN_DIR, 
        'data/waymo-200.jsonl')
    # with jsonlines.open(json_file, 'w') as f:
    #     for file in files:
    #         with open(file, 'r', encoding='utf-8') as f2:
    #             text = f2.read()
    #         img = file[:-3] + 'jpg'
    #         f.write({ 'image': img, 'text': text })

    dataset = Dataset.from_json(json_file)
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    dataset = DatasetDict(train=dataset)
    dataset.save_to_disk(os.path.join(
        MAIN_DIR, 
        'data/waymo-preprocessed'))

# make_prompt()
save_to_disk()