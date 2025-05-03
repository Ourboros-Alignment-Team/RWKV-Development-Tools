import os, json

from gevent import monkey
monkey.patch_all()

os.environ["WORKING_MODE"] = "infer_service"

from config import global_config

infer_config = global_config.infer_service_config

from bottle import route, run, request, response, Bottle
from utils import inference_service_app
from utils.message_manager import Conversation, cList
import torch

host = global_config.server_config.infer.host
port = global_config.server_config.infer.port

infer_app = inference_service_app.InferenceAPP()

app = Bottle()



def test():
    try:
        # 设置响应为JSON格式
        response.content_type = "application/json"
        # 返回服务器状态
        return json.dumps({"status": "running", "code": 200})
    except Exception as e:
        # 捕捉异常并返回错误信息
        response.status = 500  # Internal Server Error
        return json.dumps({"status": "error", "message": str(e)})


def regist_state_id_service():
    req = dict(request.json)
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    return {"access_token": infer_app.regist_state_id(load_dir)}


def remove_state_id_service():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    infer_app.remove_state_id(id)
    return {"message": "success"}


def reset_state_id():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    infer_app.states_pool[id] = torch.load(load_dir) if load_dir else None
    return {"message": "success"}


def load_weight_service():
    req = dict(request.json)
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    infer_app.model.load_weights(load_dir)
    print(f"load weights from {load_dir}.")
    return {"message": "success"}


def load_state_service():
    req = dict(request.json)
    load_dir = req.get("load_dir")  # 从请求中获取 tokens
    state_id = req.get("state_id")
    infer_app.load_state(state_id, load_dir)
    return {"message": "success"}


def save_state_service():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    to_dir = req.get("to_dir", None)  # 从请求中获取 tokens
    infer_app.save_state(id, to_dir)
    return {"message": "success"}


def copy_state_service():
    req = dict(request.json)
    from_id = req.get("from_access_token")  # 从请求中获取 tokens
    to_id = req.get("to_access_token")  # 从请求中获取 tokens
    infer_app.copy_state(from_id, to_id)
    return {"message": "success"}


def infer_batch_service():
    req = dict(request.json)
    tokens_list = req.get("tokens_list")  # 从请求中获取 tokens
    state_idx_list = req.get("state_idx_list", None)  # 可选的 state_idx
    save_cache_dir = req.get("save_cache_dir", None)
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)  # 可选的 state_idx
    state = None
    if state_idx_list:
        for state_idx in state_idx_list:
            if state is None:
                state = infer_app.states_pool.get(
                    state_idx, device=infer_app.model.device
                )
            else:
                state += infer_app.states_pool.get(
                    state_idx, device=infer_app.model.device
                )
    # 调用 infer 函数进行推理
    out, states, latent_out = infer_app.infer_batch(
        tokens_list, state, latent_output=True, save_cache_dir=save_cache_dir
    )
    if save_to_now_state_idx:
        infer_app.states_pool[save_to_now_state_idx] = states
    return {"message": "success"}


def infer_service():
    req = dict(request.json)
    conversations = req.get("conversations")  # 从请求中获取 tokens
    conversations = cList.from_dicts(conversations)
    state_idx = req.get("state_idx", None)  # 可选的 state_idx
    save_logits = req.get("save_logits", True)
    save_folder = req.get("save_folder")
    save_name = req.get("save_name")
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)  # 可选的 state_idx
    state = (
        infer_app.states_pool.get(
            state_idx, device=infer_app.model.device
        )
        if state_idx
        else None
    )

    tokens = conversations.to_tokens(infer_app.tokenizer.encode)[0]
    # 调用 infer 函数进行推理
    (
        logits,
        state,
    ) = infer_app.model.infer(tokens, state)

    if save_to_now_state_idx:
        infer_app.states_pool[save_to_now_state_idx] = state
    if save_logits:
        torch.save(logits.cpu(), os.path.join(save_folder, f"{save_name}.logits"))
    return {"message": "success"}


def infer_tokens_service():
    req = dict(request.json)
    tokens = req.get("tokens")  # 从请求中获取 tokens
    state_idx = req.get("state_idx", None)  # 可选的 state_idx
    save_logits = req.get("save_logits", True)
    save_folder = req.get("save_folder")
    save_name = req.get("save_name")
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)  # 可选的 state_idx
    state = (
        infer_app.states_pool.get(
            state_idx, device=infer_app.model.device
        )
        if state_idx
        else None
    )
    # 调用 infer 函数进行推理
    (
        logits,
        state,
    ) = infer_app.block_infer(tokens, state)
    if save_to_now_state_idx:
        infer_app.states_pool[save_to_now_state_idx] = state
    if save_logits:
        torch.save(logits.cpu(), os.path.join(save_folder, f"{save_name}.logits"))

    return {"message": "success"}




def chat_task():
    req = dict(request.json)
    conversations = req.get("conversations")
    resp_start_with_role = req.get("resp_start_with_role")
    resp_start_with_str = req.get("resp_start_with_str")
    stop_with_tokens = req.get("stop_with_tokens")
    stop_supp_tokens = req.get("stop_supp_tokens", [])
    temp = req.get("temp", 1.0)
    top_p = req.get("top_p", 0.7)
    presence_penalty = req.get("presence_penalty", 0.2)
    frequency_penalty = req.get("frequency_penalty", 0.2)
    decay_penalty = req.get("decay_penalty", 0.9961)
    use_now_state_idx = req.get("use_now_state_idx", None)
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)
    max_resp_len = req.get("max_resp_len", 512)
    format_constrain_str = req.get("format_constrain_str", None)
    token_ban = req.get("token_ban", [])

    start_with_tokens = (
        (cList.from_dicts(conversations).to_tokens(infer_app.tokenizer.encode)[0]
        if conversations
        else [])
        + global_config.role[resp_start_with_role]["prefix"]
        + (
            infer_app.tokenizer.encode(resp_start_with_str)
            if resp_start_with_str
            else []
        )
    )
    return infer_app.run_chat_task(
        start_with_tokens=start_with_tokens,
        stop_with_tokens=stop_with_tokens,
        stop_supp_tokens=stop_supp_tokens,
        temp=temp,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        penalty_decay=decay_penalty,
        constaraint_str=format_constrain_str,
        use_now_state_idx=use_now_state_idx,
        save_to_now_state_idx=save_to_now_state_idx,
        max_resp_len=max_resp_len,
        token_ban=token_ban,
    )
    
def generate_task():
    req = dict(request.json)
    start_with_tokens = req.get("start_with_tokens")
    stop_with_tokens = req.get("stop_with_tokens")
    stop_supp_tokens = req.get("stop_supp_tokens", [])
    temp = req.get("temp", 1.0)
    top_p = req.get("top_p", 0.7)
    presence_penalty = req.get("presence_penalty", 0.2)
    frequency_penalty = req.get("frequency_penalty", 0.2)
    decay_penalty = req.get("decay_penalty", 0.9961)
    use_now_state_idx = req.get("use_now_state_idx", None)
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)
    max_resp_len = req.get("max_resp_len", 512)
    format_constrain_str = req.get("format_constrain_str", None)
    token_ban = req.get("token_ban", [])
    return infer_app.run_chat_task(
        start_with_tokens=start_with_tokens,
        stop_with_tokens=stop_with_tokens,
        stop_supp_tokens=stop_supp_tokens,
        temp=temp,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        penalty_decay=decay_penalty,
        constaraint_str=format_constrain_str,
        use_now_state_idx=use_now_state_idx,
        save_to_now_state_idx=save_to_now_state_idx,
        max_resp_len=max_resp_len,
        token_ban=token_ban,
    )


def estimate_desires_service():
    req = dict(request.json)
    # 获取请求中的 role 和 prefix
    role_prefix_pairs = req.get("role_prefix_pairs")
    start_with_conversations = req.get("start_with_conversations")
    target_tokens_list = []
    for role, prefix in role_prefix_pairs:
        target_tokens_list.append(
            global_config.role[role]["prefix"] + infer_app.tokenizer.encode(prefix)
        )

    ignore_tokens = req.get(
        "ignore_tokens", [11, 33, 261, 263, 41, 42]
    )  # 默认忽略的 tokens
    ignore_tolerance = req.get("ignore_tolerance", 2)
    use_now_state_idx = req.get("use_now_state_idx", None)
    start_with_tokens = cList.from_dicts(start_with_conversations).to_tokens(
        infer_app.tokenizer.encode
    )[0]

    hit = infer_app.estimate_desires(
        target_tokens_list=target_tokens_list,
        start_with_tokens=start_with_tokens,
        ignore_tokens=ignore_tokens,
        ignore_tolerance=ignore_tolerance,
        use_now_state_idx=use_now_state_idx,
    )

    # 返回推断结果
    return {"hit": hit}


app.route("/test", method="GET", callback=test)
app.route("/regist_state_id", method="POST", callback=regist_state_id_service)
app.route("/remove_state_id", method="POST", callback=remove_state_id_service)
app.route("/reset_state_id", method="POST", callback=reset_state_id)
app.route("/load_weight", method="POST", callback=load_weight_service)
app.route("/load_state", method="POST", callback=load_state_service)
app.route("/save_state", method="POST", callback=save_state_service)
app.route("/copy_state", method="POST", callback=copy_state_service)
app.route("/infer_batch", method="POST", callback=infer_batch_service)
app.route("/infer", method="POST", callback=infer_service)
app.route("/infer_tokens", method="POST", callback=infer_tokens_service)
app.route("/estimate_desires", method="POST", callback=estimate_desires_service)
app.route("/chat_task", method="POST", callback=chat_task)
app.route("/generate_task", method="POST", callback=generate_task)


def run_server():
    # 启动 Bottle 服务器
    run(app, host=host, port=port, server="paste")


run_server()
