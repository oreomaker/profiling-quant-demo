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