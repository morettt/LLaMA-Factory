import json

def parse_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    conversations = []
    temp_dialogue = []
    for line in lines:
        if line.strip() == "":
            if temp_dialogue:
                # Process the collected dialogue
                process_dialogue(temp_dialogue, conversations)
                temp_dialogue = []
        else:
            temp_dialogue.append(line.strip())

    # Process the last dialogue if any
    if temp_dialogue:
        process_dialogue(temp_dialogue, conversations)

    return conversations

def process_dialogue(dialogue_lines, conversations):
    dialogue_block = {"messages": [], "label": None}
    for line in dialogue_lines:
        if line.startswith("用户："):
            dialogue_block["messages"].append({"content": line[3:], "role": "user"})
        elif line.startswith("助手："):
            dialogue_block["messages"].append({"content": line[3:], "role": "assistant"})
        elif line.startswith("反馈："):
            dialogue_block["label"] = True if line[3:] == "true" else False
    if dialogue_block["messages"]:
        conversations.append(dialogue_block)

def save_as_json(conversations, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)

# 设置输入输出文件路径
input_file_path = '/root/LLaMA-Factory/数据集全自动处理/KTO数据集.txt'
output_file_path = '/root/LLaMA-Factory/data/kto.json'

# 解析对话并保存为JSON
conversations = parse_conversations(input_file_path)
save_as_json(conversations, output_file_path)

print("Conversion complete! Output saved to:", output_file_path)
