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
import shutil
import threading
import time
from typing import TYPE_CHECKING, Generator

from ..extras.packages import is_gradio_available
from .model_list import MODELS


if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def _get_free_space_gb(path: str) -> float:
    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    return free / (2**30)


def _get_model_size_gb(model_id: str) -> float | None:
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        files = api.get_model_files(model_id, recursive=True)
        total_bytes = sum(f.get("Size", 0) for f in files)
        return total_bytes / (2**30)
    except Exception:
        return None


def _get_folder_size_gb(folder: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (2**30)


def _update_model_choices(series: str) -> "gr.Dropdown":
    choices = [name for _, name in MODELS.get(series, [])]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _get_model_id(series: str, model_name: str) -> str:
    for mid, mname in MODELS.get(series, []):
        if mname == model_name:
            return mid
    return ""


def _check_space(download_path: str, series: str, model_name: str) -> str:
    if not download_path.strip():
        return "请填写下载路径。"
    model_id = _get_model_id(series, model_name)
    if not model_id:
        return "请先选择模型。"
    free_gb = _get_free_space_gb(download_path)
    lines = [f"路径：{download_path}", f"当前剩余空间：{free_gb:.1f} GB", ""]
    lines.append("正在查询模型大小（可能需要几秒）...")
    model_gb = _get_model_size_gb(model_id)
    if model_gb is not None:
        lines.append(f"模型 [{model_name}] 大小：{model_gb:.1f} GB")
        if free_gb >= model_gb:
            lines.append("✅ 空间充足，可以下载。")
        else:
            lines.append(f"❌ 空间不足，还需 {model_gb - free_gb:.1f} GB。")
    else:
        lines.append("⚠️ 无法获取模型大小，请确认空间后手动下载。")
    return "\n".join(lines)


def _download_model(download_path: str, series: str, model_name: str) -> Generator[str, None, None]:
    if not download_path.strip():
        yield "请填写下载路径。"
        return
    model_id = _get_model_id(series, model_name)
    if not model_id:
        yield "请先选择模型。"
        return

    yield f"准备下载：{model_name}\nModel ID：{model_id}\n下载路径：{download_path}\n"

    free_gb = _get_free_space_gb(download_path)
    yield f"当前剩余空间：{free_gb:.1f} GB\n正在查询模型大小...\n"

    model_gb = _get_model_size_gb(model_id)
    if model_gb is not None:
        yield f"模型大小：{model_gb:.1f} GB\n"
        if free_gb < model_gb:
            yield f"❌ 空间不足！需要 {model_gb:.1f} GB，当前只有 {free_gb:.1f} GB，已取消下载。"
            return
        yield "空间充足，开始下载...\n"
    else:
        yield "⚠️ 无法确认模型大小，继续下载...\n"

    local_dir = os.path.join(download_path, model_name)
    result = {"done": False, "error": None, "path": None}

    def _do_download():
        try:
            from modelscope import snapshot_download
            result["path"] = snapshot_download(model_id, local_dir=local_dir)
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    threading.Thread(target=_do_download, daemon=True).start()

    while not result["done"]:
        current_gb = _get_folder_size_gb(local_dir) if os.path.exists(local_dir) else 0.0
        if model_gb:
            pct = min(current_gb / model_gb * 100, 99.0)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            yield f"[{bar}] {pct:.1f}%\n已下载：{current_gb:.2f} GB / {model_gb:.1f} GB"
        else:
            yield f"下载中... 已下载 {current_gb:.2f} GB"
        time.sleep(2)

    if result["error"]:
        yield f"❌ 下载失败：{result['error']}"
    else:
        final_gb = _get_folder_size_gb(local_dir)
        yield f"✅ 下载完毕！\n大小：{final_gb:.2f} GB\n路径：{result['path']}"


def _list_downloaded_models(download_path: str) -> list[list]:
    if not download_path.strip() or not os.path.exists(download_path):
        return []
    rows = []
    for name in sorted(os.listdir(download_path)):
        full = os.path.join(download_path, name)
        if os.path.isdir(full) and not name.startswith("."):
            size = _get_folder_size_gb(full)
            rows.append([name, f"{size:.2f} GB"])
    return rows


def _refresh_models(download_path: str) -> tuple:
    rows = _list_downloaded_models(download_path)
    names = [r[0] for r in rows]
    return rows, gr.Dropdown(choices=names, value=None)


def _delete_model(download_path: str, model_name: str | None) -> tuple:
    if not model_name:
        return _list_downloaded_models(download_path), gr.Dropdown(choices=[]), "请先选择要删除的模型。"
    folder = os.path.join(download_path, model_name)
    if not os.path.exists(folder):
        rows = _list_downloaded_models(download_path)
        return rows, gr.Dropdown(choices=[r[0] for r in rows]), f"路径不存在：{folder}"
    shutil.rmtree(folder)
    rows = _list_downloaded_models(download_path)
    return rows, gr.Dropdown(choices=[r[0] for r in rows], value=None), f"✅ 已删除：{model_name}"


def create_extra_tab() -> dict[str, "Component"]:
    series_choices = list(MODELS.keys())
    first_series = series_choices[0]
    first_models = [name for _, name in MODELS[first_series]]

    gr.Markdown("## 模型下载")

    with gr.Row():
        series_dd = gr.Dropdown(choices=series_choices, value=first_series, label="模型系列", scale=1)
        model_dd = gr.Dropdown(choices=first_models, value=first_models[0], label="选择模型", scale=2)

    download_path = gr.Textbox(value="/root/autodl-tmp", label="下载路径")

    with gr.Row():
        check_btn = gr.Button("检查磁盘空间")
        download_btn = gr.Button("开始下载", variant="primary")

    status_box = gr.Textbox(label="状态", interactive=False, lines=8)

    gr.Markdown("---\n### 已下载的模型")

    refresh_btn = gr.Button("刷新列表")
    model_table = gr.Dataframe(
        headers=["模型名称", "占用空间"],
        datatype=["str", "str"],
        interactive=False,
        label="已下载列表",
    )

    with gr.Row():
        delete_dd = gr.Dropdown(choices=[], label="选择要删除的模型", scale=3)
        delete_btn = gr.Button("删除", variant="stop", scale=1)

    delete_status = gr.Textbox(label="操作结果", interactive=False, lines=1)

    # 事件绑定
    series_dd.change(fn=_update_model_choices, inputs=series_dd, outputs=model_dd)
    check_btn.click(fn=_check_space, inputs=[download_path, series_dd, model_dd], outputs=status_box)
    download_btn.click(fn=_download_model, inputs=[download_path, series_dd, model_dd], outputs=status_box)
    refresh_btn.click(fn=_refresh_models, inputs=download_path, outputs=[model_table, delete_dd])
    delete_btn.click(fn=_delete_model, inputs=[download_path, delete_dd], outputs=[model_table, delete_dd, delete_status])

    return dict(
        extra_series=series_dd,
        extra_model=model_dd,
        extra_download_path=download_path,
        extra_status=status_box,
        extra_model_table=model_table,
        extra_delete_dd=delete_dd,
        extra_delete_status=delete_status,
    )
