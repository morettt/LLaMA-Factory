import json

def load_json(filename):
    """加载JSON文件"""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, filename):
    """保存数据到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def combine_datasets(file1, file2):
    """合并两个数据集，并去除重复条目"""
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    # 创建一个字典以消除重复的数据条目
    combined_dict = {json.dumps(item, sort_keys=True): item for item in data1 + data2}
    combined_data = list(combined_dict.values())
    
    return combined_data

# 路径可能需要根据你的文件存放位置调整
train_data = "train.json"
chongshu_data = "chongshu.json"

# 合并数据集
combined_data = combine_datasets(train_data, chongshu_data)

# 保存合并后的数据集到指定路径和文件名
output_path = "/root/LLaMA-Factory/data/kuochong.json"
save_json(combined_data, output_path)
