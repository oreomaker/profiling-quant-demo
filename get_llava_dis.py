# get photo info from json
from collections import defaultdict
from functools import partial
import gc
import json


def get_photo_info(json_file):
    with open(json_file) as f:
        data = json.load(f)
    photo_info = []
    for photo in data:
        photo_info.append(
            {
                "image": "/mllm-vit/flick30k-10/" + str(photo["id"]) + ".jpg",
                "label": photo["text"],
            }
        )
    return photo_info


image_info = get_photo_info("/mllm-vit/flick30k-10/disc.json")


import numpy as np
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "/mllm-vit/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)


def flatten_act_dict(act_dict):
    for layer, scales in act_dict.items():
        print(layer)
        if isinstance(scales, list):
            try:
                all_acts = np.array(scales).reshape(-1)
            except ValueError:
                print('------------')
                all_acts = [np.array(scale).reshape(-1) for scale in scales]
                all_acts = np.concatenate(all_acts)

            print(type(all_acts))
            print(all_acts.shape)

            act_dict[layer] = all_acts
        else:
            act_dict[layer] = flatten_act_dict(scales)
        gc.collect()

    return act_dict


def get_act_percentage(act_dict: dict, threshold: float):
    assert 0 <= threshold <= 1
    percentage = 1 - threshold
    act_percentage = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            all_acts_flattened = scales
            percentage_index = int(len(all_acts_flattened) * percentage) - 1
            nth_percentile_value = np.partition(all_acts_flattened, percentage_index)[
                percentage_index
            ]
            act_percentage[layer] = float(nth_percentile_value)
        else:
            print(layer)
            act_percentage[layer] = get_act_percentage(scales, threshold)
    return act_percentage


def get_act_distribution_stat(act_dict):
    act_distribution = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            act_distribution[layer] = {
                "mean": float(np.mean(scales)),
                "std": float(np.std(scales)),
            }
        else:
            act_distribution[layer] = get_act_distribution_stat(scales)
    return act_distribution


act_dict = defaultdict(dict)

def stat_io_hook(m, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    if name not in act_dict or "input" not in act_dict[name]:
        act_dict[name]["input"] = []
    act_dict[name]["input"].append(x.detach().cpu().numpy())
    if isinstance(y, tuple):
        y = y[0]
    if name not in act_dict or "output" not in act_dict[name]:
        act_dict[name]["output"] = []
    act_dict[name]["output"].append(y.detach().cpu().numpy())


hooks = []
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))


print("Collecting activation scales...")
from PIL import Image

for i in range(2):
    print(image_info[i])
    raw_image = Image.open(image_info[i]["image"])
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_info[i]["label"]},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
        "cuda", torch.float16
    )
    model(**inputs)

for hook in hooks:
    hook.remove()

print("begin_flatten")
act_dict = flatten_act_dict(act_dict)
print("finish flatten")

# origin model scale
print("begin_calculate")
print("get act 0")
ori_scale = get_act_percentage(act_dict, 0)
# scale after remove top 0.1% outliers
print("get act 0.001")
top_0_1_scale = get_act_percentage(act_dict, 0.001)
# get mean and std of all scales
print("get act distribution")
all_stat = get_act_distribution_stat(act_dict)
res_dict = {"ori": ori_scale, "top_0_1": top_0_1_scale, "all_stat": all_stat}
with open('get_llava_dis.json', "w") as f:
    json.dump(res_dict, f, indent=4, ensure_ascii=False)
