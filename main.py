
import json
import os
import logging


from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from retrieval import encode
from preprocess_image import generate_detailed_caption
WORKING_DIR = "./test"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


rag = LightRAG(
    # 工作目录
    working_dir=WORKING_DIR,
    # 智谱的prompt+completion代码
    llm_model_func=zhipu_complete,
    # 模型名
    llm_model_name="glm-4-flush",  # Using the most cost/performance balance model, but you can change it here.
    # 最大线程
    llm_model_max_async=4,
    # 模型最大token数字
    llm_model_max_token_size=32768,
    # embedding模型
    embedding_func=EmbeddingFunc(
        # embedding维度
        embedding_dim=4096,  # mmembed 4096
        # 最大token数字
        max_token_size=8192,
        func=lambda texts: encode(texts),
    ),
)
# 判断文件是否已经针对图片生成过Caption
isExist = False
file_path = "test_with_details.json"
if os.path.exists(file_path):
    isExist = True
if not isExist:
    with open('test.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
else:
    with open('test_with_details.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
for key, entry in data.items():
    docs = []
    image_and_caption = []
    for txt_Facts in entry['txt_posFacts']:
        fact = txt_Facts['fact']
        docs.append(fact)
    rag.insert(docs)
    for img_Facts in entry['img_posFacts']:
        img_url = img_Facts['imgUrl']
        caption = img_Facts['caption']
        if isExist:
            detailed_caption = img_Facts['detailed_caption']
        else:
            detailed_caption = generate_detailed_caption(img_url=img_url,caption=caption)
            img_Facts['detailed_caption'] = detailed_caption
        image_and_caption.append({'img':img_url, 'txt': detailed_caption})
    rag.insertImage(image_and_caption)

if not isExist:
    with open("test_with_details.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"处理完成，新文件已保存为 test_with_details.json")
# Perform naive search
# 直接根据问题对chunk的向量进行相似度匹配，找到最接近的作为背景进行回答
print(
    rag.query("\"Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color??\"", param=QueryParam(mode="hybrid"))
)

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )
