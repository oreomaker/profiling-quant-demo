from collections import defaultdict
from functools import partial
import json
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model_id = "/mllm-vit/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda")

model.eval()
device = next(model.parameters()).device
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


def get_clip_and_scale(act_dict: dict, t01m_thre=5) -> tuple:
    """
    Get the clipped(W8A8) and no clipped(shadow linear to restore origin scale) input and output scales of the model's layers.
    """
    top_0_1 = act_dict["top_0_1"]
    ori_scale = act_dict["ori"]
    stat = act_dict["all_stat"]
    act_scale = {}
    clip_top = {}
    clip_input_num = 0
    no_clip_input_num = 0
    clip_output_num = 0
    no_clip_output_num = 0
    no_clip_input_name = []
    no_clip_output_name = []

    for i in stat:
        top_0_1_input = top_0_1[i]["input"]
        top_0_1_output = top_0_1[i]["output"]
        act_scale[i] = {}
        clip_top[i] = {}
        # layer input
        if top_0_1_input * t01m_thre > ori_scale[i]["input"]:
            clip_input_num += 1
            clip_top[i]["input"] = True
            act_scale[i]["input"] = ori_scale[i]["input"]
        else:
            no_clip_input_num += 1
            clip_top[i]["input"] = False
            act_scale[i]["input"] = top_0_1[i]["input"]
            no_clip_input_name.append(i)
        # layer output
        if top_0_1_output * t01m_thre > ori_scale[i]["output"]:
            clip_output_num += 1
            clip_top[i]["output"] = True
            act_scale[i]["output"] = ori_scale[i]["output"]
        else:
            no_clip_output_num += 1
            clip_top[i]["output"] = False
            act_scale[i]["output"] = top_0_1[i]["output"]
            no_clip_output_name.append(i)

    return_dict = {
        "t01m_thre": t01m_thre,
        "clip_input_num": clip_input_num,
        "no_clip_input_num": no_clip_input_num,
        "clip_output_num": clip_output_num,
        "no_clip_output_num": no_clip_output_num,
        "no_clip_input_name": no_clip_input_name,
        "no_clip_output_name": no_clip_output_name,
    }
    return act_scale, clip_top, return_dict


act_dict = json.load(open("/mllm-vit/profilling_act/qwen_vl_2b_dis.json"))
act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, 100)

print(f"clip input num: {return_dict['clip_input_num']}")
print(f"clip output num: {return_dict['clip_output_num']}")
print(f"no clip input num: {return_dict['no_clip_input_num']}")
for i in return_dict["no_clip_input_name"]:
    print(f"no clip input: {i}")
print(f"no clip output num: {return_dict['no_clip_output_num']}")
for i in return_dict["no_clip_output_name"]:
    print(f"no clip output: {i}")

model_dict = model.state_dict()

for i in act_scales:
    model_dict[i + ".input_scale"] = torch.tensor(act_scales[i]["input"])
    model_dict[i + ".output_scale"] = torch.tensor(act_scales[i]["output"])
    model_dict[i + ".clip_input"] = torch.tensor(clip_top[i]["input"])
    model_dict[i + ".clip_output"] = torch.tensor(clip_top[i]["output"])


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w = w.to("cuda")
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    if n_bits == 8:
        w = w.to("cpu").type(torch.int8)
    elif n_bits == 16:
        w = w.to("cpu").type(torch.int32)
    else:
        w = w.to("cpu").type(torch.int8)
    scale = scales.to("cpu").type(torch.float32)
    return w, scale


new_model = {}
for name, param in model_dict.items():
    if "vision" in name or "lm_head" in name or "visual" in name:
        new_model[name] = param
        continue
    if name.replace(".weight", "") in act_scales:
        if "head" not in name:
            layer_name = name
            new_model[layer_name], scale = quantize_weight_per_tensor_absmax(
                model_dict[layer_name], 8
            )
            new_model[layer_name + ".scale"] = scale
            # NOTE: the int8 weight used for QNN in mllm needs to be transposed
            new_model[name] = new_model[name].transpose(-2, -1)
            # print(f"Quantized {layer_name} with scale {scale}")
        else:
            new_model[name] = param
            # print(f"Copy {name}")
    elif name.replace(".bias", "") in act_scales:
        if "head" not in name:
            layer_name = name
            new_model[layer_name], scale = quantize_weight_per_tensor_absmax(
                model_dict[layer_name], 8
            )
            new_model[layer_name + ".scale"] = scale
            # print(f"Quantized {layer_name} with scale {scale}")
        else:
            new_model[name] = param
            # print(f"Copy {name}")
    else:
        new_model[name] = param
        # print(f"Copy {name}")

torch.save(new_model, "/mllm-vit/profilling_act/qwen-2b-vl-int8-llm.pth")
print(f"Model saved to /mllm-vit/profilling_act/qwen-2b-vl-int8-llm.pth")
