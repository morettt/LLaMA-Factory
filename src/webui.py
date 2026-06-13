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

import os

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from llamafactory.extras.misc import fix_proxy, is_env_enabled
from llamafactory.webui.extra_ui import create_extra_ui
from llamafactory.webui.interface import create_ui


def main():
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "6006"))

    fix_proxy(ipv6_enabled=gradio_ipv6)

    app = FastAPI()

    @app.get("/")
    async def root():
        return RedirectResponse(url="/main/")

    app = gr.mount_gradio_app(app, create_ui().queue(), path="/main", root_path="/main")
    app = gr.mount_gradio_app(app, create_extra_ui().queue(), path="/test", root_path="/test")

    print(f"Visit http://127.0.0.1:{server_port}/       — redirects to Main WebUI")
    print(f"Visit http://127.0.0.1:{server_port}/main/  — Main WebUI")
    print(f"Visit http://127.0.0.1:{server_port}/test/  — Tools")

    uvicorn.run(app, host=server_name, port=server_port)


if __name__ == "__main__":
    main()
