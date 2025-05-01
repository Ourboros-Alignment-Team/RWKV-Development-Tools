import gradio as gr
import json
import os
from functools import partial
from config import global_config

# 文件夹路径
folder_path = global_config.save_dataset_dir
history_data = {}
file_path = ""


# 加载 JSONL 文件
def load_history(file_path):
    global history_data
    with open(file_path, "r", encoding="utf-8") as f:
        history_data = [json.loads(line) for line in f]
    return history_data


# 获取当前轮次的对话内容
def get_round_data(round_key):
    global history_data
    for entry in history_data:
        if round_key in entry:
            return entry[round_key]
    return None


# 显示当前轮次的对话
def display_round(round_key, choice_index):
    round_data = get_round_data(round_key)
    if round_data is None:
        return "无数据", "", "", gr.update(maximum=0, value=0), 0, False

    choice = round_data[choice_index]
    conversation = choice["choice"]
    score = choice["score"]
    safety = choice["safety"]
    is_best = choice["is_best"]  # 判断当前项是否为最佳
    conversation_text = "\n".join(
        [
            f"{item['role']}: "
            + item["content"].replace("\\n", "\\\\n").replace("\n", "\\n")
            for item in conversation
        ]
    )
    return (
        conversation_text,
        score,
        safety,
        gr.update(maximum=len(round_data) - 1, value=choice_index),
        is_best,  # 返回是否为最佳的状态
    )


# 保存最佳选择
def save_best_choice(round_key, choice_index, conversation_text, score, safety):
    round_data = get_round_data(round_key)
    if round_data is None:
        return "无数据，无法保存", False

    for i, choice in enumerate(round_data):
        if i == choice_index:
            choice["is_best"] = True
        else:
            choice["is_best"] = False
    update_round_data(round_key, choice_index, conversation_text, score, safety)
    return "最佳选择已保存", True


# 保存修改后的数据
def save_data():
    global history_data, file_path
    new_file_path = file_path.replace(".jsonl", "_modified.jsonl")
    with open(new_file_path, "w", encoding="utf-8") as f:
        for entry in history_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return f"数据已保存到 {new_file_path}"


# 更新当前对话的内容、分数和安全性
def update_round_data(round_key, choice_index, conversation_text, score, safety):
    round_data = get_round_data(round_key)
    if round_data is None:
        return "无数据，无法更新"

    conversation_lines = conversation_text.strip().split("\n")
    conversation_lines = [
        c.replace("\\\\n", "-=|br|=-").replace("\\n", "\n").replace("-=|br|=-", "\\n")
        for c in conversation_lines
    ]
    conversation = [
        {"role": line.split(": ")[0], "content": ": ".join(line.split(": ")[1:])}
        for line in conversation_lines
    ]

    round_data[choice_index]["choice"] = conversation
    round_data[choice_index]["score"] = int(score) if score else None
    round_data[choice_index]["safety"] = int(safety) if safety else None

    return "数据已更新"


# 自定义排序函数
def custom_sort_key(key):
    parts = key.split("-")
    return [int(part) if part.isdigit() else part for part in parts]




# Gradio 界面
def create_labeling_tab():
    with gr.Blocks() as labeling_demo:
        # 动态读取文件夹中的文件
        def refresh_file_list():
            files = []
            for root, dirs, files_in_dir in os.walk(folder_path):
                for file in files_in_dir:
                    if file.endswith(".jsonl"):
                        files.append(os.path.join(root, file))
            return files

        with gr.Row():
            file_selector = gr.Dropdown(
                choices=refresh_file_list(), label="选择文件", scale=7
            )
            refresh_files_btn = gr.Button("刷新文件列表")

        # 界面组件
        round_selector = gr.Dropdown(choices=[], label="选择轮次")
        with gr.Row():
            prev_button = gr.Button("←")
            next_button = gr.Button("→")

        choice_slider = gr.Slider(
            minimum=0, maximum=0, step=1, label="选择选项", value=0
        )
        conversation_text = gr.Textbox(label="对话内容", lines=10)
        best_toggle = gr.Checkbox(
            label="是否为最佳", interactive=False
        )  # 不可编辑的Toggle
        with gr.Row():
            score_input = gr.Textbox(label="Score")
            safety_input = gr.Textbox(label="Safety")

        with gr.Row():
            save_best_btn = gr.Button("确认最佳")
            update_data_btn = gr.Button("更新数据")
            save_data_btn = gr.Button("保存到文件(强化学习)")
        message_output = gr.Textbox(label="消息", interactive=False)

        def load_file(file_name):
            global file_path
            file_path = file_name
            load_history(file_path)
            round_keys = sorted(
                [key for entry in history_data for key in entry.keys()],
                key=custom_sort_key,
            )
            return gr.update(choices=round_keys), "文件已加载"

        def update_display(round_key, choice_index):
            return display_round(round_key, choice_index)

        def change_round(current_key, direction):
            round_keys = sorted(
                [key for entry in history_data for key in entry.keys()],
                key=custom_sort_key,
            )
            try:
                current_index = round_keys.index(current_key)
                new_index = current_index + direction
                if 0 <= new_index < len(round_keys):
                    new_key = round_keys[new_index]
                    return new_key, *display_round(new_key, 0)
            except ValueError:
                pass
            return current_key, *display_round(current_key, 0)

        # 事件绑定
        file_selector.change(
            fn=load_file,
            inputs=[file_selector],
            outputs=[round_selector, message_output],
        )
        refresh_files_btn.click(
            fn=lambda: gr.update(choices=refresh_file_list()),
            inputs=[],
            outputs=[file_selector],
        )
        round_selector.change(
            fn=partial(update_display, choice_index=0),
            inputs=[round_selector],
            outputs=[
                conversation_text,
                score_input,
                safety_input,
                choice_slider,
                best_toggle,  # 更新Toggle的显示状态
            ],
        )
        choice_slider.change(
            fn=update_display,
            inputs=[round_selector, choice_slider],
            outputs=[
                conversation_text,
                score_input,
                safety_input,
                choice_slider,
                best_toggle,  # 更新Toggle的显示状态
            ],
        )

        # 保存最佳选择
        save_best_btn.click(
            save_best_choice,
            inputs=[
                round_selector,
                choice_slider,
                conversation_text,
                score_input,
                safety_input,
            ],
            outputs=[message_output, best_toggle],
        )

        # 保存数据
        save_data_btn.click(save_data, inputs=[], outputs=[message_output])

        # 更新选项的对话内容、分数和安全性
        update_data_btn.click(
            update_round_data,
            inputs=[
                round_selector,
                choice_slider,
                conversation_text,
                score_input,
                safety_input,
            ],
            outputs=[message_output],
        )


        # 左右箭头功能
        prev_button.click(
            fn=partial(change_round, direction=-1),
            inputs=[round_selector],
            outputs=[
                round_selector,
                conversation_text,
                score_input,
                safety_input,
                choice_slider,
                best_toggle,  # 更新Toggle的显示状态
            ],
        )
        next_button.click(
            fn=partial(change_round, direction=1),
            inputs=[round_selector],
            outputs=[
                round_selector,
                conversation_text,
                score_input,
                safety_input,
                choice_slider,
                best_toggle,  # 更新Toggle的显示状态
            ],
        )

    return labeling_demo
