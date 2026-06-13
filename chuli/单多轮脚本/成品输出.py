import json
input_file = '/root/LLaMA-Factory/chuli/多轮对话处理合并.txt'
output_file = '/root/LLaMA-Factory/data/train.json'

def parse_dialogue(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    dialogues = content.strip().split('\n\n')
    parsed_data = []
    
    for dialogue in dialogues:
        lines = dialogue.split('\n')
        valid_lines = [line for line in lines if '：' in line]
        
        if not valid_lines:
            continue
            
        # 检查是否有指令格式
        if valid_lines[0].startswith('指令：'):
            # SFT格式处理
            instruction = valid_lines[0].split('：', 1)[1]
            if len(valid_lines) > 2:  # 确保有问和答
                input_text = valid_lines[1].split('：', 1)[1]
                output = valid_lines[2].split('：', 1)[1]
                
                # 处理后续的多轮对话
                history_start_index = 3
                history = []
                if len(valid_lines) > history_start_index:
                    for i in range(history_start_index, len(valid_lines) - 1, 2):
                        if i + 1 < len(valid_lines):
                            q = valid_lines[i].split('：', 1)[1]
                            a = valid_lines[i+1].split('：', 1)[1]
                            history.append([q, a])
            else:
                continue
                
            parsed_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "system": "",
                "history": history
            })
        else:
            # 原有问答对处理方式
            first_speaker, first_sentence = valid_lines[0].split('：', 1)
            
            if first_speaker != "问":
                continue
                
            instruction = first_sentence
            if len(valid_lines) > 1:
                output = valid_lines[1].split('：', 1)[1]
                history_start_index = 2
            else:
                output = ""
                history_start_index = 1
                
            history = [[valid_lines[i].split('：')[1], valid_lines[i+1].split('：')[1]] 
                      for i in range(history_start_index, len(valid_lines) - 1, 2)]
            
            parsed_data.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "system": "",
                "history": history
            })
    
    return parsed_data

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

parsed_data = parse_dialogue(input_file)
save_to_json(parsed_data, output_file)
print(f"成功保存到： {output_file}")