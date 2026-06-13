import yaml
import threading
from openai import OpenAI
import concurrent.futures

with open('config.yaml','r',encoding='utf-8') as f:
    config = yaml.safe_load(f)

client = OpenAI(api_key=config['api_key'],base_url=config['api_base'])
write_lock = threading.Lock()
TURNS = config.get('turns', 2)


def generate_ai_response(history):
    messages = [{'role':'system','content':config['system_prompt']}] + history
    response = client.chat.completions.create(
        model=config['model'],
        messages=messages
    )
    return (response.choices[0].message.content or '').strip().replace('\n', ' ')


def generate_user_followup(history):
    conversation = '\n'.join(
        f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}"
        for m in history
    )
    messages = [
        {'role':'system','content':'你的任务是根据对话内容，生成用户接下来自然会说的一句话。要口语化、简短，只输出用户说的话，不加任何前缀。'},
        {'role':'user','content':f'对话内容：\n{conversation}\n\n用户接下来说：'}
    ]
    response = client.chat.completions.create(
        model=config['model'],
        messages=messages
    )
    return (response.choices[0].message.content or '').strip().replace('\n', ' ')


def manage_line(line):
    if '问：' not in line:
        return

    ask_content = line.split('问：')[1].strip()
    if not ask_content:
        return

    try:
        history = [{'role':'user','content':ask_content}]
        output_parts = [f'问：{ask_content}\n']

        for turn in range(TURNS):
            ai_response = generate_ai_response(history)
            if not ai_response:
                print(f'[轮{turn+1}答] AI返回空，跳过本条')
                return
            history.append({'role':'assistant','content':ai_response})
            output_parts.append(f'答：{ai_response}\n')
            print(f'[轮{turn+1}答] {ai_response}')

            if turn < TURNS - 1:
                user_followup = generate_user_followup(history)
                if not user_followup:
                    print(f'[轮{turn+2}问] 生成追问为空，跳过本条')
                    return
                history.append({'role':'user','content':user_followup})
                output_parts.append(f'问：{user_followup}\n')
                print(f'[轮{turn+2}问] {user_followup}')

        with write_lock:
            with open(config['output_file'],'a',encoding='utf-8') as f:
                f.writelines(output_parts)
                f.write('\n')

    except Exception as e:
        print(f'[错误] {ask_content[:20]}... -> {e}')


def main():
    with open(config['ask_dir'],'r',encoding='utf-8') as f:
        lines = f.readlines()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as chuli:
        chuli.map(manage_line,lines)

if __name__ == '__main__':
    main()
