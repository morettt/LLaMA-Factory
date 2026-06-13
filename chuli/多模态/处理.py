import json

# 读取输入文件
input_path = '/root/LLaMA-Factory/数据集全自动处理/多模态数据集.txt'
output_path = '/root/LLaMA-Factory/data/mllm.json'

# 创建结果列表
result = []

# 读取文件并处理
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 如果是图片路径行
        if line.startswith('图片路径'):
            # 更健壮的分割方式
            image_path = line.replace('图片路径：', '').replace('图片路径:', '').strip()
            
            # 获取问题和答案（跳过空行）
            while i + 1 < len(lines) and not lines[i+1].strip().startswith('问'):
                i += 1
            question = lines[i+1].strip().replace('问:', '').replace('问：', '').strip()
            
            while i + 2 < len(lines) and not lines[i+2].strip().startswith('答'):
                i += 1
            answer = lines[i+2].strip().replace('答:', '').replace('答：', '').strip()
            
            # 创建数据项
            item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "images": [
                    image_path
                ]
            }
            
            result.append(item)
            i += 3  # 移动到下一组数据
        i += 1

# 写入JSON文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"处理完成！共处理了 {len(result)} 组对话")