import time, json, sys
sys.path.insert(0, "/lab/haoq_lab/cse12313326/MLLM/HiDe/Hide/Qwen2.5-FALSS")

import torch
from transformers import AutoProcessor
from modeling_qwen2_5_vl_re_infer import Qwen2_5_VLForConditionalGeneration
from Get_box import get_inputs, messages2out, messages2att, from_img_and_att_get_cropbox
from inference import once_infer
from PIL import Image

# 加载样本
with open("/lab/haoq_lab/cse12313326/MLLM/HiDe/Hide/Qwen2.5-FALSS/Vstar.json") as f:
    sample = json.load(f)[0]
print(f"样本: {sample['question'][:80]}...")

# 加载模型
model_path = "/lab/haoq_lab/cse12313326/MLLM/models/Qwen2.5-VL-7B-Instruct"
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True,
    min_pixels=256*28*28, max_pixels=16384*28*28)
print(f"模型加载: {time.time()-t0:.1f}s\n")

# ===== 测试1: 单次 get_yes_no_confidence =====
print("=== 测试1: get_yes_no_confidence 单次耗时 ===")
from Get_box import get_yes_no_confidence
test_img = Image.open(sample["image_path"]).convert("RGB")
crop = test_img.crop((100, 100, 200, 200))
for i in range(3):
    t0 = time.time()
    conf = get_yes_no_confidence(model, processor, crop,
        "Is there a flag clearly visible in this image? Answer yes or no.")
    print(f"  第{i+1}次: conf={conf:.3f}, {time.time()-t0:.2f}s")

# ===== 测试2: 完整 once_infer 各阶段计时 =====
print("\n=== 测试2: 完整 once_infer ===")
img_url = [sample["image_path"]]
ques = sample["question"].replace("\nAnswer with the option's letter from the given choices directly.","")

messages = [{"role": "user", "content": []}]
for img in img_url:
    messages[-1]["content"].append({"type": "image", "image": img})

t_total = time.time()

# 阶段1: 基线回答
t0 = time.time()
messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices directly."})
text, ii, vi, inputs, vk = get_inputs(messages, processor, model)
ori_ans, _ = messages2out(model, processor, inputs)
print(f"阶段1 基线回答: {time.time()-t0:.1f}s, 答案={ori_ans[0]}")

# 阶段2: HiDe 推理
t0 = time.time()
outputs = once_infer(model, processor, sample, messages, img_url, img_url, ques, [3], [0.7])
print(f"阶段2 HiDe推理总耗时: {time.time()-t0:.1f}s")
for s in [3]:
    for t in [0.7]:
        print(f"  HiDe_s{s}_t{t} 答案={outputs[str(s)][str(t)][1][0]}")

print(f"\n总耗时: {time.time()-t_total:.1f}s")
