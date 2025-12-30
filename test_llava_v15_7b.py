
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO


current_dir = os.path.dirname(os.path.abspath(__file__))

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



device = "cuda"
conv_mode = "llava_v1"
model_path = "/home/liying/Documents/llava-v1.5-7b"
image_file = os.path.join(current_dir, "Qwen2.5-VL-7B-Instruct_Based/tests/input_image.png")
qs = "Please describe this image in detail."
temperature = -1
top_p = None
num_beams = 1
max_new_tokens = 512

# deco hyparams
use_deco = True
alpha = 0.5
threshold_top_p = 0.9
threshold_top_k = 20
early_exit_layers=[i for i in range(20, 29, 1)]


model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path,
                                                                       model_base=None,
                                                                       model_name=model_name,
                                                                       device=device,
                                                                       device_map=device)

if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
else:
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

image = Image.open(image_file)
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    with torch.no_grad():
        output_dict = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(device),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_deco=use_deco,
            alpha=alpha,
            threshold_top_p=threshold_top_p,
            threshold_top_k=threshold_top_k,
            early_exit_layers=early_exit_layers,
            return_dict=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            stopping_criteria=[stopping_criteria]
        )

output_ids = output_dict.sequences
input_token_len = input_ids.shape[1]
outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
outputs = outputs.strip()
