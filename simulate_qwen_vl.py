"""
This file is a simulation of the inference process of a model that has been quantized using the quantization functions in the `quantization_simulation.py` file. 
The model is loaded from the path specified in the `model_name` argument, and the activation scales are loaded from the path specified in the `scale_file` argument. The `t01m_clip_threshold` argument specifies the threshold for clipping the activations. 
The model is quantized using the specified `model_type` argument, which determines the quantization function to be used. The quantized model is then used to generate an example based on the provided prompt.
"""

import argparse
import json
from PIL import Image

import torch
from utils.get_input_output_scales import get_clip_and_scale
from utils.quantization_simulation import (
    quantize_qwen2vl_like,
)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json


def get_photo_info(json_file):
    with open(json_file) as f:
        data = json.load(f)
    photo_info = []
    for photo in data:
        photo_info.append(
            {
                "image": "/workspace/flick30k-10/" + str(photo["id"]) + ".jpg",
                "label": photo["text"],
            }
        )
    return photo_info


image_info = get_photo_info("/workspace/flick30k-10/disc.json")

model_id = '/workspace/Qwen2-VL-2B-Instruct'

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='/workspace/Qwen2-VL-2B-Instruct')
    parser.add_argument("--scale_file", type=argparse.FileType("r"), default='/workspace/profilling_act/assets/qwen2_vl_2b_dis-0001.json')
    parser.add_argument("--t01m_clip_threshold", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map="cuda"
    )
    act_dict = json.load(open(args.scale_file.name))

    act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, args.t01m_clip_threshold)

    print(f"clip input num: {return_dict['clip_input_num']}")
    print(f"clip output num: {return_dict['clip_output_num']}")
    print(f"no clip input num: {return_dict['no_clip_input_num']}")
    for i in return_dict["no_clip_input_name"]:
        print(f"no clip input: {i}")
    print(f"no clip output num: {return_dict['no_clip_output_num']}")
    for i in return_dict["no_clip_output_name"]:
        print(f"no clip output: {i}")
    
    q_model = quantize_qwen2vl_like(model, act_scales, layer_clip=clip_top)

    processor = AutoProcessor.from_pretrained(model_id)

    # raw_image = Image.open(image_info[0]["image"])
    raw_image = Image.open("/workspace/flick30k-10/bus.png")
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[raw_image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to('cuda')

    with torch.no_grad():
        output = q_model.generate(
            **inputs, max_new_tokens=100, do_sample=False, top_p=None, top_k=None
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
