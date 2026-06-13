import json

def process_and_transform_file(input_path, output_path):
    try:
        all_dialogues = []
        messages = []
        
        with open(input_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("前缀："):
                    messages.append({
                        "role": "system",
                        "content": line[3:].strip()
                    })
                elif line.startswith("问："):
                    messages.append({
                        "role": "user",
                        "content": line[2:].strip()
                    })
                elif line.startswith("答："):
                    messages.append({
                        "role": "assistant",
                        "content": line[2:].strip()
                    })
                elif line == "" and messages:
                    all_dialogues.append({
                        "messages": messages
                    })
                    messages = []
            
            # 处理最后一组对话
            if messages:
                all_dialogues.append({
                    "messages": messages
                })
        
        # 写入格式化的 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(all_dialogues, output_file, ensure_ascii=False, indent=2)
                
        print(f"转换完成，已保存至：{output_path}")
        
    except Exception as e:
        print(f"处理文件时发生错误：{e}")

# 输入输出路径
input_path = "/root/LLaMA-Factory/数据集全自动处理/新文件.txt"
output_path = "/root/LLaMA-Factory/data/duihua.json"  # 注意这里改成了 .json 后缀

# 执行转换
process_and_transform_file(input_path, output_path)