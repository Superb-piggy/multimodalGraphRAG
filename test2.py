from io import BytesIO
from PIL import Image
import requests

# Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Extreme_sports.jpg/800px-Extreme_sports.jpg", stream=True).raw)
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Extreme_sports.jpg/800px-Extreme_sports.jpg"
response = requests.get(url, headers=headers)
image_data = BytesIO(response.content)
Image.open(image_data)

Image.open(BytesIO(requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Extreme_sports.jpg/800px-Extreme_sports.jpg",headers={'User-Agent': 'Mozilla/5.0'}).content))