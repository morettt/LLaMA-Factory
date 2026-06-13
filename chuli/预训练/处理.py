import json

# 读取输入文件
input_path = '/root/LLaMA-Factory/数据集全自动处理/预训练数据集.txt'
output_path = '/root/LLaMA-Factory/data/pt.json'

# 创建结果列表
result = []

# 读取文件并处理
with open(input_path, 'r', encoding='utf-8') as f:
    # 读取所有文本
    content = f.read()
    # 用空行分割文本
    documents = content.split('\n\n')
    # 去除空文档和"答："
    documents = [doc.strip().replace('答：', '') for doc in documents if doc.strip()]
    
    # 转换为所需格式
    for doc in documents:
        result.append({"text": doc})

# 写入JSON文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"处理完成！共处理了 {len(result)} 条数据")