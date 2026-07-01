import yaml
import threading
import concurrent.futures
import os
from datetime import datetime
from openai import OpenAI

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 每次运行生成新文件，按年月日时分秒命名
_output_dir = os.path.dirname(os.path.abspath(config['output_file']))
_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
config['output_file'] = os.path.join(_output_dir, f'{_timestamp}.txt')

client = OpenAI(api_key=config['api_key'], base_url=config['api_base'])
write_lock = threading.Lock()
print_lock = threading.Lock()
TURNS = config.get('turns', 2)
counter = {'done': 0, 'success': 0, 'fail': 0}


def generate_ai_response(history):
    messages = [{'role': 'system', 'content': config['system_prompt']}] + history
    response = client.chat.completions.create(model=config['model'], messages=messages)
    return (response.choices[0].message.content or '').strip().replace('\n', ' ')


def generate_user_followup(history):
    conversation = '\n'.join(
        f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in history
    )
    messages = [
        {'role': 'system', 'content': '你的任务是根据对话内容，生成用户接下来自然会说的一句话。要口语化、简短，只输出用户说的话，不加任何前缀。'},
        {'role': 'user', 'content': f'对话内容：\n{conversation}\n\n用户接下来说：'},
    ]
    response = client.chat.completions.create(model=config['model'], messages=messages)
    return (response.choices[0].message.content or '').strip().replace('\n', ' ')


def manage_line(args):
    idx, line, total = args
    if '问：' not in line:
        counter['done'] += 1
        return

    ask_content = line.split('问：')[1].strip()
    if not ask_content:
        counter['done'] += 1
        return

    try:
        history = [{'role': 'user', 'content': ask_content}]
        output_parts = [f'问：{ask_content}\n']
        log_lines = [f'\n[{idx}/{total}]', f'问：{ask_content}']

        for turn in range(TURNS):
            ai_response = generate_ai_response(history)
            if not ai_response:
                counter['done'] += 1
                counter['fail'] += 1
                return
            history.append({'role': 'assistant', 'content': ai_response})
            output_parts.append(f'答：{ai_response}\n')
            log_lines.append(f'答：{ai_response}')

            if turn < TURNS - 1:
                user_followup = generate_user_followup(history)
                if not user_followup:
                    counter['done'] += 1
                    counter['fail'] += 1
                    return
                history.append({'role': 'user', 'content': user_followup})
                output_parts.append(f'问：{user_followup}\n')
                log_lines.append(f'问：{user_followup}')

        with write_lock:
            with open(config['output_file'], 'a', encoding='utf-8') as f:
                f.writelines(output_parts)
                f.write('\n')

        with print_lock:
            print('\n'.join(log_lines))
            print(f'✅ 第{idx}条完成')

        counter['success'] += 1

    except Exception as e:
        with print_lock:
            print(f'❌ 第{idx}条失败：{ask_content[:20]}... -> {e}')
        counter['fail'] += 1
    finally:
        counter['done'] += 1


def main():
    with open(config['ask_dir'], 'r', encoding='utf-8') as f:
        lines = [l for l in f.readlines() if '问：' in l]

    total = len(lines)
    print(f'共 {total} 条，并发线程数：{config.get("workers", 6)}')

    args = [(i + 1, line, total) for i, line in enumerate(lines)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('workers', 6)) as ex:
        ex.map(manage_line, args)

    print(f'\n蒸馏完成！成功：{counter["success"]}  失败：{counter["fail"]}  输出：{config["output_file"]}')


if __name__ == '__main__':
    main()
