import json

# 定义转换函数
def format_dialogues(file_path, output_file):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]  # 过滤空行

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):  # 确保有足够的行来形成一个完整的单元
            break
        
        if '：' in lines[i] and '：' in lines[i+1] and '：' in lines[i+2]:
            instruction = lines[i].split('：')[1]
            chosen = lines[i+1].split('：')[1]
            rejected = lines[i+2].split('：')[1]

            entry = {
                "instruction": instruction,
                "input": "",
                "chosen": chosen,
                "rejected": rejected
            }
            data.append(entry)
        else:
            print(f"Warning: Skipping malformed lines at index {i}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 指定文件路径
file_path = '/root/LLaMA-Factory/数据集全自动处理/DPO数据集.txt'
output_file = '/root/LLaMA-Factory/data/dpo.json'

# 调用函数
format_dialogues(file_path, output_file)
