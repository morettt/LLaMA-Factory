import requests

url = 'https://api.oick.cn/api/random'
params = {'type': 'pc'}  # pc为电脑壁纸，pe为手机壁纸

response = requests.get(url, params=params)

with open('wallpaper.jpg', 'wb') as f:
    f.write(response.content)

print('图片已保存为wallpaper.jpg')