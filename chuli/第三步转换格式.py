import json

# 定义文件路径
file_path = '/root/LLaMA-Factory/chuli/内容回复2.txt'
output_file_path = '/root/LLaMA-Factory/data/train2.json'

def convert_file_to_llm_format(file_path):
    llm_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):  # 遍历到倒数第二行，避免索引越界
            if lines[i].startswith('问：') and lines[i+1].startswith('答：'):
                instruction = lines[i].split('问：', 1)[1].strip()
                output = lines[i+1].split('答：', 1)[1].strip()
                llm_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
    return llm_data

def save_llm_data_to_file(llm_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(llm_data, file, ensure_ascii=False, indent=4)

llm_formatted_data = convert_file_to_llm_format(file_path)
save_llm_data_to_file(llm_formatted_data, output_file_path)

print(f'数据已成功保存到文件：{output_file_path}')
