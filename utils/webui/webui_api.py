import json
import time
from utils import llms_api_chatbot
import re, os

api_base_input_list = [
    "",
    "",
    "",
    "",
    "",
]
api_key_input_list = [
    "",
    "",
    "",
    "",
    "",
]
api_model_input_list = [
    "",
    "",
    "",
    "",
    "",
]


def api_infer(
    history_messages: list,
    api_base: str,
    api_key: str,
    api_model: str,
    temp: float = 1,
    top_p: float = 0.7,
    presence_penalty: float = 0.2,
    frequency_penalty: float = 0.2,
):
    bot = llms_api_chatbot.OpenAPIChatBot(api_base, api_key, api_model)
    iterator = bot.stream_chat(
        history_messages,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
    )
    return iterator


api_chat_history = []


def hist_to_cb_hist(api_chat_history: list):
    result = []
    for hist in api_chat_history:
        content = hist["content"]
        # 查找<think>标签
        import re

        think_parts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)

        if think_parts:
            # 对每个找到的<think>部分处理
            remaining_content = content
            for think in think_parts:
                # 删除这部分内容
                remaining_content = remaining_content.replace(
                    f"<think>{think}</think>", ""
                )

                # 插入思考过程
                think_item = {
                    "role": hist["role"],
                    "content": think.strip(),
                    "metadata": {"title": "思考过程", "id": 0, "status": "pending"},
                }
                result.append(think_item)

            # 添加剩余内容
            if remaining_content.strip():
                hist_copy = hist.copy()
                hist_copy["content"] = remaining_content.strip()
                result.append(hist_copy)
        else:
            # 如果没有<think>标签,直接添加原始项
            result.append(hist)

    return result


def api_chat(
    usr_name: str,
    bot_name: str,
    content: str,
    api_base: str,
    api_key: str,
    api_model: str,
    temp: float = 1,
    top_p: float = 0.7,
    presence_penalty: float = 0.2,
    frequency_penalty: float = 0.2,
):
    api_chat_history.append({"role": "user", "content": f"{usr_name}: {content}"})
    for txt, full_txt in api_infer(
        api_chat_history,
        api_base,
        api_key,
        api_model,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
    ):
        yield hist_to_cb_hist(
            api_chat_history + [{"role": "assistant", "content": f"{bot_name}: {full_txt}"}]
        )
    api_chat_history.append({"role": "assistant", "content": f"{bot_name}: {full_txt}"})


def api_add(role, content):
    api_chat_history.append({"role": role, "content": content})
    return hist_to_cb_hist(api_chat_history)


def api_back():
    global api_chat_history
    # 找到最后一个用户消息的索引
    last_user_index = -1
    for i in reversed(range(len(api_chat_history))):
        if api_chat_history[i]["role"] == "user":
            last_user_index = i
            break

    # 删除该用户消息之后的所有内容
    api_chat_history = api_chat_history[:last_user_index]
    return hist_to_cb_hist(api_chat_history)


def api_regenerate(
    bot_name: str,
    api_base: str,
    api_key: str,
    api_model: str,
    temp: float = 1,
    top_p: float = 0.7,
    presence_penalty: float = 0.2,
    frequency_penalty: float = 0.2,
):
    # 删除最后一个assistant回复
    if api_chat_history and api_chat_history[-1]["role"] == "assistant":
        api_chat_history.pop()

    # 重新生成时直接复用最后一个用户输入
    iterator = api_infer(
        api_chat_history,
        api_base,
        api_key,
        api_model,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
    )

    # 流式生成并实时更新历史记录
    for txt, full_txt in iterator:
        yield hist_to_cb_hist(
            api_chat_history + [{"role": "assistant", "content": f"{bot_name}: {full_txt}"}]
        )
    api_chat_history.append({"role": "assistant", "content": f"{bot_name}: {full_txt}"})


def api_reset():
    api_chat_history.clear()
    return hist_to_cb_hist(api_chat_history)


def api_listen(
    bot_name: str,
    api_base: str,
    api_key: str,
    api_model: str,
    temp: float = 1,
    top_p: float = 0.7,
    presence_penalty: float = 0.2,
    frequency_penalty: float = 0.2,
):
    # 直接基于当前历史生成回复
    iterator = api_infer(
        api_chat_history,
        api_base,
        api_key,
        api_model,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
    )

    # 流式生成并实时更新历史记录
    for txt, full_txt in iterator:
        yield hist_to_cb_hist(
            api_chat_history + [{"role": "assistant", "content": f"{bot_name}: {full_txt}"}]
        )
    api_chat_history.append({"role": "assistant", "content": f"{bot_name}: {full_txt}"})


# ========================================


def save_api_hist(save_folder):
    # Ensure save_folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Get the current time for filename
    timestamp = time.strftime("%y-%m-%d-%H-%M-%S")
    filename = f"{timestamp}.json"
    file_path = os.path.join(save_folder, filename)

    # Save chat history as JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(api_chat_history, f, ensure_ascii=False, indent=4)

    return f"Chat history saved to {file_path}"


def load_api_hist(load_dir):
    global api_chat_history
    # Load JSON file from load_dir
    if not os.path.exists(load_dir):
        return f"File not found: {load_dir}"

    with open(load_dir, "r", encoding="utf-8") as f:
        api_chat_history = json.load(f)

    return f"Chat history loaded from {load_dir}"
