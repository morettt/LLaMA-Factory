import json
import os
import requests
from openai import OpenAI
from transformers.utils.versions import require_version

require_version("openai>=1.5.0", "修复方法：pip install openai>=1.5.0")

def get_bing_daily_image() -> str:
    """获取必应每日图片并保存"""
    url = "https://api.oick.cn/api/bing"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = 'bing_daily.jpg'
            with open(filename, 'wb') as f:
                f.write(response.content)
            return f"图片已成功保存为 {filename}"
        else:
            return f"获取图片失败，状态码: {response.status_code}"
    except Exception as e:
        return f"获取图片时发生错误: {str(e)}"

def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 6006)),
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_bing_daily_image",
                "description": "当用户明确要求获取必应每日图片时，调用此功能下载图片",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]

    tool_map = {"get_bing_daily_image": get_bing_daily_image}

    messages = [
        {"role": "system", "content": """你是一个友好的AI助手。你可以进行日常对话，也可以帮助用户获取必应每日图片。
只有当用户明确表示想要获取必应图片时，才使用图片获取功能。
保持自然的对话风格，不要主动提及或推销图片获取功能。"""}
    ]
    
    print("你好！我是AI助手，很高兴和你聊天。输入'退出'结束对话。")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("再见！")
            break
            
        messages.append({"role": "user", "content": user_input})

        try:
            result = client.chat.completions.create(
                messages=messages, 
                model="test", 
                tools=tools,
                temperature=0.7
            )
            assistant_message = result.choices[0].message
            
            if assistant_message.tool_calls is not None:
                messages.append(assistant_message)
                
                for tool_call in assistant_message.tool_calls:
                    function_call = tool_call.function
                    name = function_call.name
                    
                    if name in tool_map:
                        tool_result = tool_map[name]()
                        messages.append({
                            "role": "tool",
                            "content": tool_result
                        })
                
                result = client.chat.completions.create(messages=messages, model="test", tools=tools)
                print("助手:", result.choices[0].message.content)
            else:
                messages.append(assistant_message)
                print("助手:", assistant_message.content)
                
        except Exception as e:
            print(f"抱歉，发生了一些错误: {str(e)}")
            if messages:
                messages.pop()

if __name__ == "__main__":
    main()