import base64
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests


# # Each query needs to be accompanied by an corresponding instruction describing the task.
# task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question"}

# instruction = task_name_to_instruct['example']
# queries = [
#     {'txt': 'are judo throws allowed in wrestling?'}, 
#     {'txt': 'how to become a radiology technician in michigan?'},
# ]

# # No instruction needed for retrieval passages
# passages = [
#     {'txt': "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force."},
#     {'txt': "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."},
# ]

# 设备映射字典
device_map = {}
device_map["image_newline"] = f"cuda:{0}"

device_list_a = [
    "multi_modal_projector.linear_1.bias", "multi_modal_projector.linear_1.weight",
    "multi_modal_projector.linear_2.bias", "multi_modal_projector.linear_2.weight",
    "vision_tower.vision_model.embeddings.class_embedding",
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "vision_tower.vision_model.post_layernorm.bias", "vision_tower.vision_model.post_layernorm.weight",
    "vision_tower.vision_model.pre_layrnorm.bias", "vision_tower.vision_model.pre_layrnorm.weight"
]

for layer in range(24):
    prefix = f"vision_tower.vision_model.encoder.layers.{layer}"
    device_list_a.extend([
        f"{prefix}.layer_norm1.bias", f"{prefix}.layer_norm1.weight",
        f"{prefix}.layer_norm2.bias", f"{prefix}.layer_norm2.weight",
        f"{prefix}.mlp.fc1.bias", f"{prefix}.mlp.fc1.weight",
        f"{prefix}.mlp.fc2.bias", f"{prefix}.mlp.fc2.weight",
        f"{prefix}.self_attn.k_proj.bias", f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.q_proj.bias", f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.v_proj.bias", f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.out_proj.bias", f"{prefix}.self_attn.out_proj.weight",
    ])

device_list_a.extend(["language_model.embed_tokens.weight"])

for layer in range(16):
    prefix = f"language_model.layers.{layer}"
    device_list_a.extend([
        f"{prefix}.input_layernorm.weight", f"{prefix}.mlp.down_proj.weight",
        f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.o_proj.weight",
    ])

device_list_b = ["language_model.norm.weight"]
for layer in range(16, 32):
    prefix = f"language_model.layers.{layer}"
    device_list_b.extend([
        f"{prefix}.input_layernorm.weight", f"{prefix}.mlp.down_proj.weight",
        f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.o_proj.weight",
    ])
device_list_c = []
device_list_c.extend([
    "latent_attention_model.cross_attend_blocks.0.fn.to_q.weight", "latent_attention_model.cross_attend_blocks.0.fn.to_kv.weight",
    "latent_attention_model.cross_attend_blocks.0.fn.to_out.weight", "latent_attention_model.cross_attend_blocks.0.norm.bias",
    "latent_attention_model.cross_attend_blocks.0.norm.weight", "latent_attention_model.cross_attend_blocks.0.norm_context.bias",
    "latent_attention_model.cross_attend_blocks.0.norm_context.weight", "latent_attention_model.cross_attend_blocks.1.fn.net.0.bias",
    "latent_attention_model.cross_attend_blocks.1.fn.net.0.weight", "latent_attention_model.cross_attend_blocks.1.fn.net.2.bias",
    "latent_attention_model.cross_attend_blocks.1.fn.net.2.weight", "latent_attention_model.cross_attend_blocks.1.norm.bias",
    "latent_attention_model.cross_attend_blocks.1.norm.weight", "latent_attention_model.latents"
])

for key in device_list_a:
    device_map[key] = f"cuda:{0}"
for key in device_list_b:
    device_map[key] = f"cuda:{1}"
for key in device_list_c:
    device_map[key] = f"cuda:{3}"


# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/MM-Embed', trust_remote_code=True,device_map=device_map)


# max_length = 4096
# query_embeddings = model.encode(queries, is_query=True, instruction=instruction, max_length=max_length)['hidden_states']
# passage_embeddings = model.encode(passages, max_length=max_length)['hidden_states']
# # compute relevance scores
# scores = (query_embeddings @ passage_embeddings.T) * 100
# print(scores.tolist())



async def encode(text=None, image=None, instruction=None, max_length=4096):
    """
    传入文本或图片，返回对应的 embedding。

    参数：
        model: 预训练的 MM-Embed 模型
        text: 字符串或字符串列表
        image: 图片路径或 PIL.Image 对象
        instruction: 任务指令（仅文本查询需要）
        max_length: 最大序列长度，默认 4096

    返回：
        embedding: 计算出的 embedding 向量 (NumPy 数组)
    """
    if text is not None and image is None:
        # 文本编码
        input_data = [{'txt': text}] if isinstance(text, str) else [{'txt': t} for t in text]
        embeddings = model.encode(input_data, is_query=(instruction is not None), instruction=instruction, max_length=max_length)['hidden_states']
    
    elif image is not None and text is None:
        # 图片编码
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        input_data = [{'img': image}]
        embeddings = model.encode(input_data, max_length=max_length)['hidden_states']
    
    elif text is not None and image is not None:
        # 文本 + 图片联合编码
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        input_data = [{'txt': text, 'img': image}]
        embeddings = model.encode(input_data, max_length=max_length)['hidden_states']
    
    else:
        raise ValueError("必须传入 text 或 image 中至少一个参数")
    
    # 确保返回值是 NumPy 数组
    if isinstance(embeddings, torch.Tensor):  # 如果是 PyTorch 张量
        embeddings = embeddings.cpu().numpy()  # 移动到 CPU 并转换为 NumPy 数组
    elif not isinstance(embeddings, np.ndarray):  # 如果不是 NumPy 数组
        embeddings = np.array(embeddings)  # 转换为 NumPy 数组
    
    return embeddings