import base64
import json
from zhipuai import ZhipuAI
import os
import requests
from PIL import Image
from io import BytesIO

prompt = """
You are provided with a URL of an image and its existing caption. The current caption describes the main subject of the image and may include relationships between entities in the scene. Your task is to generate a more detailed and precise caption for the image while preserving the relationships described in the existing caption.
The new caption should:
1.Retain the relationships and interactions mentioned in the current caption.
2.Add additional descriptive details about the image, including:
    A.Background elements (e.g., landscape, weather, environment).
    B.People (if present): their actions, clothing, or other notable features.
    C.Geographical location (if identifiable).
    D.Architecture or structures in the image.
    E.Any other notable features or objects in the scene.
3.Focus solely on factual and descriptive content. Avoid including any subjective opinions, interpretations, or assumptions.
### Example:
- **Input:**
  - Current Caption: "A man is walking his dog near a lake."
- **Output:**
  - New Caption: "A man wearing a blue jacket is walking his brown dog near a calm lake surrounded by tall pine trees. The sky is clear with a few fluffy white clouds, and there is a wooden bench nearby."

Current Caption: {current_caption}
Generate a new caption that adheres to the above guidelines.
"""

def download_image_and_save(url, save_file, save_format="JPEG"):
    """
    从 URL 下载图片并以指定格式保存
    :param url: 图片 URL
    :param save_file: 保存文件路径
    :param save_format: 保存的图片格式，默认为 JPEG
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        try:
            image = Image.open(BytesIO(response.content))
            # 修改为正确的格式
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(save_file, format=save_format.upper())
            print(f"Image saved successfully: {save_file}")
        except Exception as e:
            raise Exception(f"Failed to process image: {e}")
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    return save_file



def generate_detailed_caption(img_url, caption):
    download_image_and_save(img_url, 'image.jpg')
    with open("image.jpg", 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    client = ZhipuAI(api_key=api_key) 
    response = client.chat.completions.create(
        model="glm-4v-plus",  
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(current_caption=caption)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

