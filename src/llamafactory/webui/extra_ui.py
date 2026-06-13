# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


def create_extra_ui() -> "gr.Blocks":
    with gr.Blocks(title="LLaMA Factory - Tools") as demo:
        gr.Markdown("# Tools\n在这里添加你的小工具。")

        with gr.Tab("示例工具"):
            gr.Markdown("这是一个示例工具页，按需替换。")
            input_text = gr.Textbox(label="输入")
            output_text = gr.Textbox(label="输出", interactive=False)
            btn = gr.Button("运行")
            btn.click(fn=lambda x: f"收到：{x}", inputs=input_text, outputs=output_text)

    return demo
