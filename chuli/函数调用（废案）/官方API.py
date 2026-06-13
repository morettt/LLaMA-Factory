import json
import os
import requests
from typing import Sequence
from openai import OpenAI

def download_bing_image() -> str:
    """
    下载必应每日图片
    """
    url = "https://api.oick.cn/api/bing"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = 'bing_daily.jpg'
            with open(filename, 'wb') as f:
                f.write(response.content)
            return f"图片已成功保存为 {filename}"
        else:
            return f"下载失败，状态码: {response.status_code}"
    except Exception as e:
        return f"发生错误: {str(e)}"

def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 6006)),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "download_bing_image",
                "description": "从必应下载今日图片",
                "parameters": {
                    "type": "object",
                    "properties": {},  # 清空所有参数
                    "required": []
                }
            }
        }
    ]

    tool_map = {"download_bing_image": download_bing_image}
    
    messages = [
        {
            "role": "system",
            "content": "你是一个友好的AI助手。你可以进行日常对话，也可以帮助用户获取必应每日图片。只有当用户明确表示想要获取必应图片时，才使用图片获取功能。保持自然的对话风格，不要主动提及或推销图片获取功能。"
        }
    ]
    
    print("你好！我是AI助手。输入 'exit' 结束对话。")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'exit':
            print("再见！")
            break
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            result = client.chat.completions.create(messages=messages, model="test", tools=tools)
            
            if hasattr(result.choices[0].message, 'tool_calls') and result.choices[0].message.tool_calls is not None:
                messages.append(result.choices[0].message)
                tool_call = result.choices[0].message.tool_calls[0].function
                print("正在执行操作...")
                
                name = tool_call.name
                arguments = json.loads(tool_call.arguments) if tool_call.arguments else {}
                
                if name in tool_map:
                    tool_result = tool_map[name]()  # 不传递任何参数
                    messages.append({"role": "tool", "content": str(tool_result)})
                
                result = client.chat.completions.create(messages=messages, model="test", tools=tools)
            
            ai_response = result.choices[0].message.content
            messages.append({"role": "assistant", "content": ai_response})
            print("\nAI:", ai_response)
            
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    main()