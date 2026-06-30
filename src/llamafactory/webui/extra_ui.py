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

import base64
import concurrent.futures
import json
import os
import re
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
    return gr.Dropdown(choices=choices, value=None)


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


# key = model_name，每个 slot 独立跟踪一个下载任务
_active_downloads: dict[str, dict] = {}
_downloads_lock = threading.Lock()


def _get_combined_status() -> str:
    with _downloads_lock:
        slots = dict(_active_downloads)
    if not slots:
        return ""
    parts = []
    for model_name, slot in slots.items():
        parts.append(f"─── {model_name} ───\n{slot['status']}")
    return "\n\n".join(parts)


def _any_active() -> bool:
    with _downloads_lock:
        return any(slot["active"] for slot in _active_downloads.values())


def _run_full_download_thread(model_id: str, local_dir: str, model_name: str, available_gb: float, reserved_gb: float):
    """后台线程：查询大小 → 检查空间 → 下载，全程更新 _active_downloads[model_name]['status']。"""
    model_gb = _get_model_size_gb(model_id)
    with _downloads_lock:
        if model_name not in _active_downloads:
            return
        _active_downloads[model_name]["model_gb"] = model_gb

    if model_gb is not None:
        if available_gb < model_gb:
            msg = f"❌ 空间不足！{model_name} 需要 {model_gb:.1f} GB，可用仅 {available_gb:.1f} GB"
            if reserved_gb > 0:
                msg += f"（含已预留 {reserved_gb:.1f} GB）"
            with _downloads_lock:
                _active_downloads[model_name]["status"] = msg
                _active_downloads[model_name]["active"] = False
            return
        with _downloads_lock:
            _active_downloads[model_name]["status"] = f"准备下载：{model_name}\n大小：{model_gb:.1f} GB，空间充足，开始下载..."
    else:
        with _downloads_lock:
            _active_downloads[model_name]["status"] = f"准备下载：{model_name}\n⚠️ 无法确认模型大小，继续下载..."

    dl_result = {"done": False, "error": None, "path": None}

    def _do_download():
        try:
            from modelscope import snapshot_download
            dl_result["path"] = snapshot_download(model_id, local_dir=local_dir)
        except Exception as e:
            dl_result["error"] = e
        finally:
            dl_result["done"] = True

    threading.Thread(target=_do_download, daemon=True).start()

    while not dl_result["done"]:
        current_gb = _get_folder_size_gb(local_dir) if os.path.exists(local_dir) else 0.0
        with _downloads_lock:
            mgb = _active_downloads.get(model_name, {}).get("model_gb")
        if mgb:
            pct = min(current_gb / mgb * 100, 100.0)
            bar = "█" * 20 if pct >= 100.0 else "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            status = (
                f"[{bar}] 正在校验/整理文件，请稍候..."
                if pct >= 100.0
                else f"[{bar}] {pct:.1f}%\n已下载：{current_gb:.2f} GB / {mgb:.1f} GB"
            )
        else:
            status = f"下载中... 已下载 {current_gb:.2f} GB"
        with _downloads_lock:
            if model_name in _active_downloads:
                _active_downloads[model_name]["status"] = status
        time.sleep(2)

    with _downloads_lock:
        if model_name in _active_downloads:
            if dl_result["error"]:
                _active_downloads[model_name]["status"] = f"❌ 下载失败：{dl_result['error']}"
            else:
                final_gb = _get_folder_size_gb(local_dir)
                _active_downloads[model_name]["status"] = f"✅ 下载完毕！\n大小：{final_gb:.2f} GB\n路径：{dl_result['path']}"
            _active_downloads[model_name]["active"] = False


def _start_download(download_path: str, series: str, model_name: str) -> str:
    """非 streaming：立即注册下载任务并返回，进度由 Timer 轮询更新。"""
    if not download_path.strip():
        return "请填写下载路径。"
    model_id = _get_model_id(series, model_name)
    if not model_id:
        return "请先选择模型。"

    with _downloads_lock:
        if model_name in _active_downloads and _active_downloads[model_name]["active"]:
            return _get_combined_status()
        reserved_gb = sum(
            slot["model_gb"] for slot in _active_downloads.values()
            if slot["active"] and slot["model_gb"]
        )

    local_dir = os.path.join(download_path, model_name)
    free_gb = _get_free_space_gb(download_path)
    available_gb = free_gb - reserved_gb

    with _downloads_lock:
        _active_downloads[model_name] = {
            "active": True,
            "model_gb": None,
            "local_dir": local_dir,
            "status": (
                f"准备下载：{model_name}\nModel ID：{model_id}\n"
                f"磁盘剩余：{free_gb:.1f} GB"
                + (f"（已为其他下载预留 {reserved_gb:.1f} GB，实际可用 {available_gb:.1f} GB）" if reserved_gb > 0 else "")
                + "\n正在查询模型大小..."
            ),
        }

    threading.Thread(
        target=_run_full_download_thread,
        args=(model_id, local_dir, model_name, available_gb, reserved_gb),
        daemon=True,
    ).start()

    return _get_combined_status()


def _poll_status():
    """给 gr.Timer 用：有下载时返回最新状态，否则不更新组件。"""
    with _downloads_lock:
        has_downloads = bool(_active_downloads)
    if not has_downloads:
        return gr.update()
    return _get_combined_status()


def _resume_download_stream() -> Generator[str, None, None]:
    """页面刷新后由 demo.load 调用，若有任意下载仍在进行则重新接入进度流。"""
    if not _any_active():
        return
    while _any_active():
        yield _get_combined_status()
        time.sleep(2)
    yield _get_combined_status()


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


def _build_model_table_html(rows: list) -> str:
    if not rows:
        return "<p style='color:#6b7280;font-size:14px;padding:4px 0'>暂无已下载的模型</p>"
    html_rows = []
    for name, size in rows:
        safe = name.replace("'", "\\'")
        html_rows.append(
            f"<tr>"
            f"<td style='padding:5px 12px;border-bottom:1px solid var(--border-color-primary)'>"
            f"{name}"
            f"<button onclick=\"navigator.clipboard.writeText('{safe}').then(()=>{{this.textContent='✓';setTimeout(()=>this.textContent='⎘',1000)}})\" "
            f"style='margin-left:6px;cursor:pointer;background:none;border:none;color:#9ca3af;font-size:12px;padding:0 2px;vertical-align:middle' title='复制'>⎘</button>"
            f"</td>"
            f"<td style='padding:5px 12px;border-bottom:1px solid var(--border-color-primary);color:#6b7280'>{size}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
        "<thead><tr style='background:var(--table-even-background-fill)'>"
        "<th style='text-align:left;padding:6px 12px;border-bottom:1px solid var(--border-color-primary);font-weight:600'>模型名称</th>"
        "<th style='text-align:left;padding:6px 12px;border-bottom:1px solid var(--border-color-primary);font-weight:600'>占用空间</th>"
        "</tr></thead>"
        "<tbody>" + "".join(html_rows) + "</tbody>"
        "</table>"
    )


def _refresh_models(download_path: str) -> tuple:
    rows = _list_downloaded_models(download_path)
    names = [r[0] for r in rows]
    return _build_model_table_html(rows), gr.Dropdown(choices=names, value=None)


def _delete_model(download_path: str, model_name: str | None) -> tuple:
    if not model_name:
        rows = _list_downloaded_models(download_path)
        return _build_model_table_html(rows), gr.Dropdown(choices=[]), "请先选择要删除的模型。"
    folder = os.path.join(download_path, model_name)
    if not os.path.exists(folder):
        rows = _list_downloaded_models(download_path)
        return _build_model_table_html(rows), gr.Dropdown(choices=[r[0] for r in rows]), f"路径不存在：{folder}"
    shutil.rmtree(folder)
    rows = _list_downloaded_models(download_path)
    return _build_model_table_html(rows), gr.Dropdown(choices=[r[0] for r in rows], value=None), f"✅ 已删除：{model_name}"


_DISTIL_CONFIG = os.path.join("数据集蒸馏房", "distil_config.json")


def _load_distil_config() -> dict:
    try:
        with open(_DISTIL_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_distil_config(**kwargs) -> None:
    config = _load_distil_config()
    config.update({k: v for k, v in kwargs.items() if v is not None and v != ""})
    os.makedirs("数据集蒸馏房", exist_ok=True)
    with open(_DISTIL_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)


def _fetch_models(api_key: str, api_base: str) -> "gr.Dropdown":
    if not api_key.strip() or not api_base.strip():
        return gr.Dropdown(choices=[], value=None)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip(), base_url=api_base.strip())
        models = sorted([m.id for m in client.models.list()])
        _save_distil_config(api_key=api_key.strip(), api_base=api_base.strip())
        return gr.Dropdown(choices=models, value=models[0] if models else None)
    except Exception:
        return gr.Dropdown(choices=[], value=None)


_distil_stop = threading.Event()
_user_navigated = False


def _stop_distil() -> str:
    _distil_stop.set()
    return "⏹ 已发送停止信号，等待当前批次完成..."


_LINES_PER_PAGE = 30


def _read_output(output_file: str) -> str:
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _get_page(output_file: str, page_idx: int) -> tuple:
    content = _read_output(output_file)
    lines = content.split("\n") if content else []
    total_pages = max(1, (len(lines) + _LINES_PER_PAGE - 1) // _LINES_PER_PAGE)
    page_idx = max(0, min(page_idx, total_pages - 1))
    start = page_idx * _LINES_PER_PAGE
    page_content = "\n".join(lines[start:start + _LINES_PER_PAGE])
    return page_content, f"第 {page_idx + 1} 页 / 共 {total_pages} 页", page_idx


def _prev_page(output_file: str, page_idx: int) -> tuple:
    global _user_navigated
    _user_navigated = True
    return _get_page(output_file, page_idx - 1)


def _next_page(output_file: str, page_idx: int) -> tuple:
    global _user_navigated
    _user_navigated = True
    return _get_page(output_file, page_idx + 1)


def _run_distil(
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    turns: int,
    input_file: str,
    output_file: str,
    workers: int,
) -> Generator[tuple, None, None]:
    global _user_navigated
    _distil_stop.clear()
    _user_navigated = False

    if not api_key.strip():
        yield "请填写 API Key。", "", "第 1 页 / 共 1 页", 0
        return
    if not os.path.exists(input_file):
        yield f"输入文件不存在：{input_file}", "", "第 1 页 / 共 1 页", 0
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key.strip(), base_url=api_base.strip())

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if "问：" in l]

    total = len(lines)
    if total == 0:
        yield "输入文件中没有找到「问：」格式的内容。", "", "第 1 页 / 共 1 页", 0
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    open(output_file, "w", encoding="utf-8").close()

    write_lock = threading.Lock()
    counter = {"done": 0, "success": 0, "fail": 0}

    def generate_ai_response(history):
        messages = [{"role": "system", "content": system_prompt}] + history
        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip().replace("\n", " ")

    def generate_user_followup(history):
        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in history
        )
        messages = [
            {"role": "system", "content": "你的任务是根据对话内容，生成用户接下来自然会说的一句话。要口语化、简短，只输出用户说的话，不加任何前缀。"},
            {"role": "user", "content": f"对话内容：\n{conversation}\n\n用户接下来说："},
        ]
        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip().replace("\n", " ")

    def process_line(line):
        if _distil_stop.is_set():
            counter["done"] += 1
            return
        ask_content = line.split("问：")[1].strip()
        if not ask_content:
            counter["done"] += 1
            return
        try:
            history = [{"role": "user", "content": ask_content}]
            parts = [f"问：{ask_content}\n"]
            for turn in range(turns):
                if _distil_stop.is_set():
                    return
                ai_resp = generate_ai_response(history)
                if not ai_resp:
                    return
                history.append({"role": "assistant", "content": ai_resp})
                parts.append(f"答：{ai_resp}\n")
                if turn < turns - 1:
                    followup = generate_user_followup(history)
                    if not followup:
                        return
                    history.append({"role": "user", "content": followup})
                    parts.append(f"问：{followup}\n")
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.writelines(parts)
                    f.write("\n")
            counter["success"] += 1
        except Exception:
            counter["fail"] += 1
        finally:
            counter["done"] += 1

    result = {"done": False}

    def _run():
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(process_line, lines)
        result["done"] = True

    threading.Thread(target=_run, daemon=True).start()

    yield f"开始蒸馏，共 {total} 条，并发 {workers} 线程...\n", gr.update(), gr.update(), gr.update(), gr.update(visible=False)
    while not result["done"]:
        pct = counter["done"] / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        status = f"[{bar}] {pct:.1f}%\n已处理：{counter['done']}/{total}  成功：{counter['success']}  失败：{counter['fail']}"
        if _distil_stop.is_set():
            status += "\n⏹ 停止中..."
        if not _user_navigated:
            page_content, page_info_val, page_idx = _get_page(output_file, 9999)
            yield status, page_content, page_info_val, page_idx, gr.update(visible=False)
        else:
            yield status, gr.update(), gr.update(), gr.update(), gr.update(visible=False)
        time.sleep(2)

    final_status = f"✅ 蒸馏完成！\n成功：{counter['success']}  失败：{counter['fail']}\n输出文件：{output_file}"
    if _distil_stop.is_set():
        final_status = f"⏹ 已停止。\n成功：{counter['success']}  失败：{counter['fail']}\n输出文件：{output_file}"
    page_content, page_info_val, page_idx = _get_page(output_file, 9999)
    yield final_status, page_content, page_info_val, page_idx, gr.update(visible=True)


def create_distil_tab() -> dict[str, "Component"]:
    cfg = _load_distil_config()

    saved_key = cfg.get("api_key", "")
    saved_base = cfg.get("api_base", "")
    saved_model = cfg.get("model", None)
    initial_choices = []
    if saved_key and saved_base:
        try:
            from openai import OpenAI
            _c = OpenAI(api_key=saved_key, base_url=saved_base)
            initial_choices = sorted([m.id for m in _c.models.list()])
        except Exception:
            pass

    gr.Markdown("## 数据集蒸馏")

    with gr.Row():
        api_key = gr.Textbox(label="API Key", value=saved_key, placeholder="sk-...", scale=2)
        api_base = gr.Textbox(label="API Base", value=saved_base, placeholder="https://...", scale=2)
        model = gr.Dropdown(label="模型", choices=initial_choices, value=saved_model if saved_model in initial_choices else None, allow_custom_value=True, scale=1)

    system_prompt = gr.Textbox(
        label="System Prompt",
        value=cfg.get("system_prompt", ""),
        placeholder="你是一个...",
        lines=3,
    )

    with gr.Row():
        turns = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="对话轮数")
        workers = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="并发线程数")

    with gr.Row():
        input_file = gr.Textbox(label="输入文件路径", value="/root/LLaMA-Factory/数据集蒸馏房/data/ordinary.txt", scale=3)
        output_file = gr.Textbox(label="输出文件路径", value="/root/LLaMA-Factory/数据集蒸馏房/data/output.txt", scale=3)

    with gr.Row():
        start_btn = gr.Button("开始蒸馏", variant="primary", scale=3)
        stop_btn = gr.Button("停止", variant="stop", scale=1)

    distil_status = gr.Textbox(label="进度", interactive=False, lines=4)
    distil_output = gr.Textbox(label="蒸馏输出", interactive=False, lines=15)

    with gr.Row():
        prev_btn = gr.Button("上一页", scale=1)
        next_btn = gr.Button("下一页", scale=1)
        page_info = gr.Textbox(value="第 1 页 / 共 1 页", interactive=False, show_label=False, scale=2)

    download_btn = gr.Button("下载数据集", variant="primary", visible=False)
    hidden_content_box = gr.Textbox(visible=False, interactive=False)

    page_state = gr.State(value=0)

    start_btn.click(
        fn=_run_distil,
        inputs=[api_key, api_base, model, system_prompt, turns, input_file, output_file, workers],
        outputs=[distil_status, distil_output, page_info, page_state, download_btn],
    )
    stop_btn.click(fn=_stop_distil, outputs=distil_status)
    prev_btn.click(fn=_prev_page, inputs=[output_file, page_state], outputs=[distil_output, page_info, page_state])
    next_btn.click(fn=_next_page, inputs=[output_file, page_state], outputs=[distil_output, page_info, page_state])

    def _read_output_for_download(fpath: str) -> str:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    download_btn.click(
        fn=_read_output_for_download,
        inputs=[output_file],
        outputs=[hidden_content_box],
    ).then(
        fn=None,
        inputs=[hidden_content_box],
        js="""(content) => {
            if (!content) return;
            const blob = new Blob([content], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'distil_output.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }"""
    )

    # 拉取模型列表 + 保存配置
    api_key.change(fn=_fetch_models, inputs=[api_key, api_base], outputs=model)
    api_base.change(fn=_fetch_models, inputs=[api_key, api_base], outputs=model)
    system_prompt.change(fn=lambda v: _save_distil_config(system_prompt=v), inputs=system_prompt)
    model.change(fn=lambda v: _save_distil_config(model=v), inputs=model)

    return dict(
        distil_api_key=api_key,
        distil_api_base=api_base,
        distil_model=model,
        distil_system_prompt=system_prompt,
        distil_turns=turns,
        distil_workers=workers,
        distil_input_file=input_file,
        distil_output_file=output_file,
        distil_status=distil_status,
        distil_output=distil_output,
        distil_page_info=page_info,
    )


def create_extra_tab() -> dict[str, "Component"]:
    series_choices = list(MODELS.keys())
    first_series = series_choices[0]
    first_models = [name for _, name in MODELS[first_series]]

    gr.Markdown("## 模型下载")

    with gr.Row():
        series_dd = gr.Dropdown(choices=series_choices, value=first_series, label="模型系列", scale=1)
        model_dd = gr.Dropdown(choices=first_models, value=None, label="选择模型（可输入关键字过滤）", scale=2, filterable=True)

    download_path = gr.Textbox(value="/root/autodl-tmp", label="下载路径")

    with gr.Row():
        check_btn = gr.Button("检查磁盘空间")
        download_btn = gr.Button("开始下载", variant="primary")

    with gr.Row():
        check_status_box = gr.Textbox(label="磁盘空间检查", interactive=False, lines=8)
        download_status_box = gr.Textbox(label="下载进度", interactive=False, lines=8)

    gr.Markdown("---\n### 已下载的模型")

    refresh_btn = gr.Button("刷新列表")
    model_table = gr.HTML(value="")

    with gr.Row():
        delete_dd = gr.Dropdown(choices=[], label="选择要删除的模型", scale=3)
        delete_btn = gr.Button("删除", variant="stop", scale=1)

    delete_status = gr.Textbox(label="操作结果", interactive=False, lines=1)

    series_dd.change(fn=_update_model_choices, inputs=series_dd, outputs=model_dd)
    check_btn.click(fn=_check_space, inputs=[download_path, series_dd, model_dd], outputs=check_status_box)
    download_btn.click(fn=_start_download, inputs=[download_path, series_dd, model_dd], outputs=download_status_box)
    refresh_btn.click(fn=_refresh_models, inputs=download_path, outputs=[model_table, delete_dd])
    delete_btn.click(fn=_delete_model, inputs=[download_path, delete_dd], outputs=[model_table, delete_dd, delete_status])
    gr.Timer(value=2).tick(fn=_poll_status, outputs=download_status_box)

    return dict(
        extra_series=series_dd,
        extra_model=model_dd,
        extra_download_path=download_path,
        extra_check_status=check_status_box,
        extra_download_status=download_status_box,
        extra_model_table=model_table,
        extra_delete_dd=delete_dd,
        extra_delete_status=delete_status,
    )


_PROCESS_MODES = {
    "SFT（单多轮对话）": {
        "input": "/root/LLaMA-Factory/数据集全自动处理/放置数据集.txt",
        "output": "/root/LLaMA-Factory/data/train.json",
    },
    "预训练": {
        "input": "/root/LLaMA-Factory/数据集全自动处理/预训练数据集.txt",
        "output": "/root/LLaMA-Factory/data/pt.json",
    },
    "KTO": {
        "input": "/root/LLaMA-Factory/数据集全自动处理/KTO数据集.txt",
        "output": "/root/LLaMA-Factory/data/kto.json",
    },
    "DPO": {
        "input": "/root/LLaMA-Factory/数据集全自动处理/DPO数据集.txt",
        "output": "/root/LLaMA-Factory/data/dpo.json",
    },
    "多模态": {
        "input": "/root/LLaMA-Factory/数据集全自动处理/多模态数据集.txt",
        "output": "/root/LLaMA-Factory/data/mllm.json",
    },
}


def _get_image_at(upload_dir: str, idx: int) -> tuple:
    if not os.path.exists(upload_dir):
        return "<p style='color:#6b7280;padding:8px'>目录不存在或暂无图片</p>", "", 0
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    files = sorted(f for f in os.listdir(upload_dir) if os.path.splitext(f)[1].lower() in exts)
    if not files:
        return "<p style='color:#6b7280;padding:8px'>暂无图片</p>", "", 0
    idx = idx % len(files)
    fname = files[idx]
    try:
        with open(os.path.join(upload_dir, fname), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        ext = os.path.splitext(fname)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        img_html = (
            f"<div style='text-align:center;padding:4px'>"
            f"<div style='height:380px;display:flex;align-items:center;justify-content:center;"
            f"border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;background:#f9fafb'>"
            f"<img src='data:image/{mime};base64,{b64}' "
            f"style='max-width:100%;max-height:380px;object-fit:contain'/>"
            f"</div>"
            f"<div style='font-size:12px;color:#6b7280;margin-top:6px'>{fname}</div>"
            f"</div>"
        )
    except Exception as e:
        img_html = f"<p style='color:red'>加载失败：{fname}（{e}）</p>"
    return img_html, f"{idx + 1} / {len(files)}", idx


def _upload_images(files, upload_dir: str, current_text: str) -> tuple:
    if not files:
        img_html, counter, idx = _get_image_at(upload_dir, 0)
        return current_text, img_html, counter, idx
    os.makedirs(upload_dir, exist_ok=True)
    new_paths = []
    for f in files:
        src = f if isinstance(f, str) else f.name
        filename = os.path.basename(src)
        dest = os.path.join(upload_dir, filename)
        shutil.copy2(src, dest)
        new_paths.append(dest)
    additions = [f"图片路径：{p}\n问：\n答：" for p in new_paths]
    new_text = current_text.rstrip("\n")
    if new_text:
        new_text += "\n\n"
    new_text += "\n\n".join(additions)
    img_html, counter, idx = _get_image_at(upload_dir, 0)
    return new_text, img_html, counter, idx


def _switch_mode(mode: str, img_dir: str = "") -> tuple:
    cfg = _PROCESS_MODES.get(mode, _PROCESS_MODES["SFT（单多轮对话）"])
    text = _load_dataset_text(cfg["input"])
    is_mllm = mode == "多模态"
    if is_mllm:
        img_html, counter, idx = _get_image_at(img_dir, 0)
    else:
        img_html, counter, idx = "", "", 0
    return text, cfg["input"], cfg["output"], gr.update(visible=is_mllm), img_html, counter, idx


def _process_sft(text: str, output_path: str) -> str:
    lines = text.splitlines()
    merged, prev_speaker, merged_line = [], None, ""
    for line in lines:
        match = re.match(r"^(问|答|提示|指令)[：:](.+)$", line.strip())
        if match:
            speaker, content = match.groups()
            if speaker == prev_speaker:
                merged_line += "。" + content
            else:
                if merged_line:
                    merged.append(f"{prev_speaker}：{merged_line}")
                prev_speaker, merged_line = speaker, content
        else:
            if merged_line:
                merged.append(f"{prev_speaker}：{merged_line}")
            merged.append(line.strip())
            prev_speaker, merged_line = None, ""
    if merged_line:
        merged.append(f"{prev_speaker}：{merged_line}")

    parsed = []
    for dialogue in "\n".join(merged).strip().split("\n\n"):
        valid = [l for l in dialogue.split("\n") if "：" in l]
        if not valid:
            continue
        if valid[0].startswith("指令："):
            instruction = valid[0].split("：", 1)[1]
            if len(valid) < 3:
                continue
            history = []
            for i in range(3, len(valid) - 1, 2):
                if i + 1 < len(valid):
                    history.append([valid[i].split("：", 1)[1], valid[i + 1].split("：", 1)[1]])
            parsed.append({"instruction": instruction, "input": valid[1].split("：", 1)[1], "output": valid[2].split("：", 1)[1], "system": "", "history": history})
        else:
            first_speaker, first_sentence = valid[0].split("：", 1)
            if first_speaker != "问":
                continue
            history = []
            for i in range(2, len(valid) - 1, 2):
                history.append([valid[i].split("：")[1], valid[i + 1].split("：")[1]])
            parsed.append({"instruction": first_sentence, "input": "", "output": valid[1].split("：", 1)[1] if len(valid) > 1 else "", "system": "", "history": history})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return f"✅ 处理完成！共生成 {len(parsed)} 条数据\n输出文件：{output_path}"


def _process_pt(text: str, output_path: str) -> str:
    documents = [d.strip().replace("答：", "") for d in text.strip().split("\n\n") if d.strip()]
    result = [{"text": d} for d in documents]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return f"✅ 处理完成！共生成 {len(result)} 条数据\n输出文件：{output_path}"


def _process_kto(text: str, output_path: str) -> str:
    conversations, temp = [], []
    for line in text.splitlines():
        if not line.strip():
            if temp:
                block = {"messages": [], "label": None}
                for l in temp:
                    if l.startswith("用户："):
                        block["messages"].append({"content": l[3:], "role": "user"})
                    elif l.startswith("助手："):
                        block["messages"].append({"content": l[3:], "role": "assistant"})
                    elif l.startswith("反馈："):
                        block["label"] = l[3:].strip().lower() == "true"
                if block["messages"]:
                    conversations.append(block)
                temp = []
        else:
            temp.append(line.strip())
    if temp:
        block = {"messages": [], "label": None}
        for l in temp:
            if l.startswith("用户："):
                block["messages"].append({"content": l[3:], "role": "user"})
            elif l.startswith("助手："):
                block["messages"].append({"content": l[3:], "role": "assistant"})
            elif l.startswith("反馈："):
                block["label"] = l[3:].strip().lower() == "true"
        if block["messages"]:
            conversations.append(block)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
    return f"✅ 处理完成！共生成 {len(conversations)} 条数据\n输出文件：{output_path}"


def _process_dpo(text: str, output_path: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = []
    for i in range(0, len(lines) - 2, 3):
        if all("：" in lines[i + j] for j in range(3)):
            data.append({"instruction": lines[i].split("：", 1)[1], "input": "", "chosen": lines[i + 1].split("：", 1)[1], "rejected": lines[i + 2].split("：", 1)[1]})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"✅ 处理完成！共生成 {len(data)} 条数据\n输出文件：{output_path}"


def _process_mllm(text: str, output_path: str) -> str:
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("图片路径"):
            image_path = line.replace("图片路径：", "").replace("图片路径:", "").strip()
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith("问"):
                i += 1
            question = lines[i + 1].strip().replace("问:", "").replace("问：", "").strip()
            while i + 2 < len(lines) and not lines[i + 2].strip().startswith("答"):
                i += 1
            answer = lines[i + 2].strip().replace("答:", "").replace("答：", "").strip()
            result.append({"conversations": [{"from": "human", "value": f"<image>{question}"}, {"from": "gpt", "value": answer}], "images": [image_path]})
            i += 3
        i += 1
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return f"✅ 处理完成！共生成 {len(result)} 组对话\n输出文件：{output_path}"


def _process_dataset(text: str, input_path: str, output_path: str, mode: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(input_path)), exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if mode == "SFT（单多轮对话）":
            return _process_sft(text, output_path)
        elif mode == "预训练":
            return _process_pt(text, output_path)
        elif mode == "KTO":
            return _process_kto(text, output_path)
        elif mode == "DPO":
            return _process_dpo(text, output_path)
        elif mode == "多模态":
            return _process_mllm(text, output_path)
        return "❌ 未知模式"
    except Exception as e:
        return f"❌ 处理失败：{e}"


def _load_dataset_text(input_path: str) -> str:
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def create_process_tab() -> dict[str, "Component"]:
    gr.Markdown("## 数据集处理")

    default_mode = "SFT（单多轮对话）"
    default_cfg = _PROCESS_MODES[default_mode]

    mode_dd = gr.Dropdown(label="训练模式", choices=list(_PROCESS_MODES.keys()), value=default_mode)

    with gr.Row():
        input_path = gr.Textbox(label="输入文件路径", value=default_cfg["input"], scale=3)
        output_path = gr.Textbox(label="输出文件路径", value=default_cfg["output"], scale=3)

    _DEFAULT_IMG_DIR = "/root/LLaMA-Factory/数据集全自动处理/图片"

    with gr.Column(visible=False) as img_section:
        gr.Markdown("### 图片管理")
        with gr.Row():
            img_upload_dir = gr.Textbox(label="图片目录", value=_DEFAULT_IMG_DIR, scale=4)
            refresh_gallery_btn = gr.Button("刷新", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                img_upload = gr.File(
                    label="上传图片（支持多选，上传后自动追加路径到下方文本）",
                    file_types=["image"],
                    file_count="multiple",
                )
            with gr.Column(scale=1):
                img_display = gr.HTML(value="")
                with gr.Row():
                    prev_img_btn = gr.Button("◄ 上一张", scale=1)
                    img_counter = gr.Textbox(value="", interactive=False, show_label=False, scale=2)
                    next_img_btn = gr.Button("下一张 ►", scale=1)
        img_idx = gr.State(value=0)

    dataset_text = gr.Textbox(label="数据集内容", value=_load_dataset_text(default_cfg["input"]), lines=20, placeholder="在此粘贴或编辑数据集...")

    process_btn = gr.Button("保存并处理", variant="primary")
    process_status = gr.Textbox(label="处理结果", interactive=False, lines=2)

    mode_dd.change(fn=_switch_mode, inputs=[mode_dd, img_upload_dir], outputs=[dataset_text, input_path, output_path, img_section, img_display, img_counter, img_idx])
    img_upload.upload(fn=_upload_images, inputs=[img_upload, img_upload_dir, dataset_text], outputs=[dataset_text, img_display, img_counter, img_idx])
    refresh_gallery_btn.click(fn=lambda d: _get_image_at(d, 0), inputs=img_upload_dir, outputs=[img_display, img_counter, img_idx])
    prev_img_btn.click(fn=lambda d, i: _get_image_at(d, i - 1), inputs=[img_upload_dir, img_idx], outputs=[img_display, img_counter, img_idx])
    next_img_btn.click(fn=lambda d, i: _get_image_at(d, i + 1), inputs=[img_upload_dir, img_idx], outputs=[img_display, img_counter, img_idx])
    process_btn.click(fn=_process_dataset, inputs=[dataset_text, input_path, output_path, mode_dd], outputs=process_status)

    return dict(
        process_mode=mode_dd,
        process_input_path=input_path,
        process_output_path=output_path,
        process_text=dataset_text,
        process_status=process_status,
    )


