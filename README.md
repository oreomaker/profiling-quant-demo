## Profiling Activation Tools

### Supported Model Type
- transformers.models.qwen2
- transformers.models.llama
- transformers.models.opt
- transformers.models.gemma
- transformers.models.phi
- transformers.models.mixtral
- transformers.models.falcon

### Examples
1. Get activation distribution config using *get_act_distribution.py*
```bash
python get_qwen1.5_act_distribution.py
```
**Caution: getting activation distribution config needs huge amount of (cpu) memory. > 100 GB Memory Volume is suggested.**

2. Use activation distribution config to predict in different threshold of clipping.
```bash
python example_run_qwen2_lambada.py
```

```bash
python export_int8_model.py --model_name ../Qwen1.5-1.8B-Chat/ --model_type qwen1 --scale_file ./qwen1.5-1.8b_act_scales_distribution.json --output_model qwen-int8-test.pth
python converter.py --input_model qwen-int8-test.pth --output_model qwen-i8-test.mllm
```

## for Qwen2 VL

```bash
# qwen2 vl activation scale profiling
python get_qwen2_vl_dis.py

# simulate int8 qwen2 vl model inference
python simulate_qwen2_vl.py

# export to int8(only linear in LLM) model
python export_int8_model.py --model_name /workspace/Qwen2-VL-2B-Instruct --model_type qwen2vl --scale_file /workspace/profilling_act/assets/qwen2_vl_2b_dis-0001.json --output_model /workspace/profilling_act/assets/qwen2-vl-i8-test.pth --t01m_clip_threshold 64

# convert to mllm format
python converter.py --input_model /workspace/profilling_act/assets/qwen2-vl-i8-test.pth --type torch --output_model assets/qwen2-vl-i8-test.mllm
```