import os

os.environ["WORKING_MODE"] = "train_service"

from gevent import monkey
monkey.patch_all()
from config import global_config

train_config = global_config.train_service_config
grpo_config = global_config.grpo


from bottle import route, run, request, response
from utils.message_manager import cList, Conversation
import torch.distributed as dist
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from utils.train_app import OnlineTrainingAPP
import json
from utils.dataset.gsm8k_dataset import GSM8KRLDataset
from utils.dataset.dataset import RLGroupDataset, RLPairDataset

app = OnlineTrainingAPP()
rank = dist.get_rank()
total_ranks = dist.get_world_size()
port = global_config.server_config.train.port_begin + rank


@route("/test", method="GET")
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


@route("/save", method="POST")
def save():
    req = dict(request.json)
    model_name = req.get("save_name", "default")
    save_train_state = req.get("save_train_state", False)
    folder = req.get("folder", None)
    app.save_weight(model_name, save_train_state, folder=folder)
    return {"message": "success"}


@route("/load_model", method="POST")
def load_model():
    req = dict(request.json)

    ckpt_dir = req.get("load_dir", "")
    lr_init = req.get("lr_init", None)
    lr_final = req.get("lr_final", None)
    warmup_steps = req.get("warmup_steps", None)

    app.load_model(
        ckpt_dir, lr_init=lr_init, lr_final=lr_final, warmup_steps=warmup_steps
    )
    return {"message": "success"}


@route("/train_single_datafolder", method="POST")
def train_single_datafolder():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_single_datafolder"))

    forder_dir = req.get("forder_dir")
    epoch = req.get("epoch", 1)
    batch_size_per_gpu = req.get("batch_size_per_gpu", 1)
    n_save_ckpt = req.get("n_save_ckpt", 1)
    multi_scale_ctx = req.get("multi_scale_ctx", train_config.model.ctx_len)
    multi_scale_alpha = req.get(
        "multi_scale_alpha", train_config.train.multi_scale_alpha
    )
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    n_save_step = req.get("n_save_step", None)
    keep_states_mode = req.get("keep_states_mode", "never")
    dataloader_workers_per_gpu = req.get("dataloader_workers_per_gpu", 4)
    begin_with_state_dir = req.get("begin_with_state_dir", None)
    use_qa_mask = req.get("use_qa_mask", False)
    lr_init = req.get("lr_init", None)
    lr_final = req.get("lr_final", None)
    warmup_steps = req.get("warmup_steps", None)

    response.content_type = "application/json"
    return app.train_from_folder(
        forder_dir=forder_dir,
        epoch=epoch,
        batch_size_per_gpu=batch_size_per_gpu,
        n_save_ckpt=n_save_ckpt,
        multi_scale_ctx=multi_scale_ctx,
        multi_scale_alpha=multi_scale_alpha,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        n_save_step=n_save_step,
        keep_states_mode=keep_states_mode,
        dataloader_workers_per_gpu=dataloader_workers_per_gpu,
        begin_with_state_dir=begin_with_state_dir,
        use_qa_mask=use_qa_mask,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        chunk_len=train_config.model.chunk_len
    )


@route("/train_from_folders", method="POST")
def train_from_folders():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_text_from_folders"))
    folder_weight_dir_list = req.get("folder_weight_dir_list", [])
    epoch = req.get("epoch", 1)
    batch_size_per_gpu = req.get("batch_size_per_gpu", 1)
    n_save_ckpt = req.get("n_save_ckpt", 1)
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    n_save_step = req.get("n_save_step", None)
    dataloader_workers_per_gpu = req.get("dataloader_workers_per_gpu", 2)
    use_qa_mask = req.get("use_qa_mask", False)
    lr_init = req.get("lr_init", None)
    lr_final = req.get("lr_final", None)
    warmup_steps = req.get("warmup_steps", None)

    response.content_type = "application/json"
    return app.train_from_folders(
        folder_weight_dir_list=folder_weight_dir_list,
        epoch=epoch,
        batch_size_per_gpu=batch_size_per_gpu,
        n_save_ckpt=n_save_ckpt,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        n_save_step=n_save_step,
        dataloader_workers_per_gpu=dataloader_workers_per_gpu,
        use_qa_mask=use_qa_mask,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
    )


@route("/train_text_from_messages", method="POST")
def train_text_from_messages():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_text_from_messages"))
    messages = req.get("messages", [])
    messages = [cList.from_dicts(msgs) for msgs in messages]
    batch_size = req.get("batch_size", 1)
    n_save_ckpt = req.get("n_save_ckpt", -1)
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    multi_scale_ctx = req.get("multi_scale_ctx", train_config.model.ctx_len)
    multi_scale_alpha = req.get(
        "multi_scale_alpha", train_config.train.multi_scale_alpha
    )
    keep_train_states = req.get("keep_train_states", False)
    use_ego_mask = req.get("use_ego_mask", False)
    ignore_ctx = req.get("ignore_ctx", False)
    save_name_last = req.get("save_name_last", "last")

    lr_init = req.get("lr_init", None)
    lr_final = req.get("lr_final", None)
    warmup_steps = req.get("warmup_steps", None)
    app.train_text_from_messages(
        messages=messages,
        batch_size=batch_size,
        n_save_ckpt=n_save_ckpt,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        multi_scale_ctx=multi_scale_ctx,
        multi_scale_alpha=multi_scale_alpha,
        keep_train_states=keep_train_states,
        use_ego_mask=use_ego_mask,
        ignore_ctx=ignore_ctx,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
    )
    app.save_weight(save_name_last, True)


@route("/train_gsm8k", method="POST")
def train_gsm8k():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_gsm8k"))
    parquet_file_path = req.get("parquet_file_path")
    ref_model_server = req.get(
        "ref_model_server",
        f"http://{global_config.server_config.infer.host}:{global_config.server_config.infer.port}",
    )
    req_role = req.get("req_role", "conversation")
    req_prefix = req.get("req_prefix", "Q: ")
    resp_role = req.get("resp_role", "response")
    resp_prefix = req.get("resp_prefix", "A: ")
    temperature = req.get("temperature", 1)
    top_p = req.get("top_p", 0.85)
    tokenizer = global_config.tokenizer_train

    rl_dataset = GSM8KRLDataset(
        parquet_file_path=parquet_file_path,
        req_role=req_role,
        req_prefix=req_prefix,
        resp_role=resp_role,
        resp_prefix=resp_prefix,
        tokenizer=tokenizer,
    )
    n_epoch = req.get("n_epoch", 1)
    n_rollout_questions = req.get("n_rollout_questions", 1)
    temperature = req.get("temperature", 1)
    top_p = req.get("top_p", 0.85)
    alpha_frequency = req.get("alpha_frequency", 0.2)
    alpha_presence = req.get("alpha_presence", 0.2)
    alpha_decay = req.get("alpha_decay", 0.9961)
    max_ctx = req.get("max_ctx", grpo_config.rollout_max_len)
    token_stop = global_config.role[resp_role]["postfix"][0:1]
    token_ban = [0]
    lr_init = req.get("lr_init", grpo_config.lr_init)
    lr_final = req.get("lr_final", grpo_config.lr_final)
    warmup_steps = req.get("warmup_steps", grpo_config.warmup_steps)
    n_save_ckpt = req.get("n_save_ckpt", 1)
    n_save_episode_ckpt = req.get("n_save_episode_ckpt", 5)
    num_rollouts = req.get("num_rollouts", grpo_config.n_rollout)
    tiny_batch_size = req.get("tiny_batch_size", grpo_config.rollout_tiny_batch)
    train_batch_size = req.get("train_batch_size", grpo_config.train_batch_size)
    n_replay_sliding_window = req.get(
        "n_replay_sliding_window", grpo_config.n_replay_sliding_window
    )
    clear_replay_on_episode = req.get(
        "clear_replay_on_episode", grpo_config.clear_replay_on_episode
    )
    n_train_each_episode = req.get(
        "n_train_each_episode", grpo_config.n_train_each_episode
    )
    clip_eps = req.get("clip_eps", grpo_config.clip_eps)
    kl_weight = req.get("kl_weight", grpo_config.kl_weight)
    grad_cp_max_norm = req.get("grad_cp_max_norm", grpo_config.grad_cp_max_norm)
    accumulate_grad = req.get("accumulate_grad", grpo_config.accumulate_grad)
    response.content_type = "application/json"
    return app.train_grpo(
        rl_dataset=rl_dataset,
        ref_model_server=ref_model_server,
        reward_func=rl_dataset.reward_func,
        rlhf_func=lambda *args: args,
        n_epoch=n_epoch,
        n_rollout_questions=n_rollout_questions,
        temperature=temperature,
        top_p=top_p,
        alpha_frequency=alpha_frequency,
        alpha_presence=alpha_presence,
        alpha_decay=alpha_decay,
        max_ctx=max_ctx,
        token_stop=token_stop,
        token_ban=token_ban,
        num_rollouts=num_rollouts,
        tiny_batch_size=tiny_batch_size,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        n_save_ckpt=n_save_ckpt,
        n_save_episode_ckpt=n_save_episode_ckpt,
        n_replay_sliding_window=n_replay_sliding_window,
        clear_replay_on_episode=clear_replay_on_episode,
        n_train_each_episode=n_train_each_episode,
        train_batch_size=train_batch_size,
        clip_eps=clip_eps,
        kl_weight=kl_weight,
        grad_cp_max_norm=grad_cp_max_norm,
        accumulate_grad=accumulate_grad,
    )


@route("/train_grpo_from_group_dataset", method="POST")
def train_grpo_from_group_dataset():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_grpo_from_group_dataset"))
    dataset_dir = req.get("dataset_dir", "")
    tokenizer = global_config.tokenizer_train
    dataset = RLGroupDataset(dataset_dir, tokenizer, None)
    ref_model_server = req.get(
        "ref_model_server",
        f"http://{global_config.server_config.infer.host}:{global_config.server_config.infer.port}",
    )
    lr_init = req.get("lr_init", grpo_config.lr_init)
    lr_final = req.get("lr_final", grpo_config.lr_final)
    warmup_steps = req.get("warmup_steps", grpo_config.warmup_steps)
    n_save_episode_ckpt = req.get("n_save_episode_ckpt", 1)
    n_replay_sliding_window = req.get(
        "n_replay_sliding_window", grpo_config.n_replay_sliding_window
    )
    clear_replay_on_episode = req.get(
        "clear_replay_on_episode", grpo_config.clear_replay_on_episode
    )
    n_train_each_episode = req.get(
        "n_train_each_episode", grpo_config.n_train_each_episode
    )
    train_batch_size = req.get("train_batch_size", grpo_config.train_batch_size)
    clip_eps = req.get("clip_eps", grpo_config.clip_eps)
    kl_weight = req.get("kl_weight", grpo_config.kl_weight)
    grad_cp_max_norm = req.get("grad_cp_max_norm", grpo_config.grad_cp_max_norm)
    accumulate_grad = req.get("accumulate_grad", grpo_config.accumulate_grad)
    response.content_type = "application/json"

    return app.train_grpo_from_group_dataset(
        group_dataset=dataset,
        ref_model_server=ref_model_server,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        n_save_episode_ckpt=n_save_episode_ckpt,
        n_replay_sliding_window=n_replay_sliding_window,
        clear_replay_on_episode=clear_replay_on_episode,
        n_train_each_episode=n_train_each_episode,
        train_batch_size=train_batch_size,
        clip_eps=clip_eps,
        kl_weight=kl_weight,
        grad_cp_max_norm=grad_cp_max_norm,
        accumulate_grad=accumulate_grad,
    )


@route("/train_grpo_from_pair_dataset", method="POST")
def train_grpo_from_pair_dataset():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_grpo_from_pair_dataset"))

    dataset_fp = req.get("dataset_fp")
    tokenizer = global_config.tokenizer_train
    n_samples_episode = req.get("n_samples_episode", 5)
    n_episodes = req.get("n_episodes", 5)
    role_system = req.get("role_system", "system")
    system_prefix = req.get("system_prefix", "system")
    role_sender = req.get("role_sender", "conversation")
    sender_prefix = req.get("sender_prefix", "Q")
    role_receiver = req.get("role_receiver", "response")
    receiver_prefix = req.get("receiver_prefix", "A")
    dataset = RLPairDataset(
        dataset_fp=dataset_fp,
        tokenizer=tokenizer,
        n_samples_episode=n_samples_episode,
        n_episodes=n_episodes,
        role_system=role_system,
        system_prefix=system_prefix,
        role_sender=role_sender,
        sender_prefix=sender_prefix,
        role_replier=role_receiver,
        replier_prefix=receiver_prefix,
    )
    ref_model_server = req.get(
        "ref_model_server",
        f"http://{global_config.server_config.infer.host}:{global_config.server_config.infer.port}",
    )
    lr_init = req.get("lr_init", grpo_config.lr_init)
    lr_final = req.get("lr_final", grpo_config.lr_final)
    warmup_steps = req.get("warmup_steps", grpo_config.warmup_steps)
    n_save_episode_ckpt = req.get("n_save_episode_ckpt", 1)
    n_replay_sliding_window = req.get(
        "n_replay_sliding_window", grpo_config.n_replay_sliding_window
    )
    clear_replay_on_episode = req.get(
        "clear_replay_on_episode", grpo_config.clear_replay_on_episode
    )
    n_train_each_episode = req.get(
        "n_train_each_episode", grpo_config.n_train_each_episode
    )
    train_batch_size = req.get("train_batch_size", grpo_config.train_batch_size)
    clip_eps = req.get("clip_eps", grpo_config.clip_eps)
    kl_weight = req.get("kl_weight", grpo_config.kl_weight)
    grad_cp_max_norm = req.get("grad_cp_max_norm", grpo_config.grad_cp_max_norm)
    accumulate_grad = req.get("accumulate_grad", grpo_config.accumulate_grad)
    response.content_type = "application/json"

    return app.train_grpo_from_group_dataset(
        group_dataset=dataset,
        ref_model_server=ref_model_server,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        n_save_episode_ckpt=n_save_episode_ckpt,
        n_replay_sliding_window=n_replay_sliding_window,
        clear_replay_on_episode=clear_replay_on_episode,
        n_train_each_episode=n_train_each_episode,
        train_batch_size=train_batch_size,
        clip_eps=clip_eps,
        kl_weight=kl_weight,
        grad_cp_max_norm=grad_cp_max_norm,
        accumulate_grad=accumulate_grad,
    )


@route("/train_grpo_online", method="POST")
def train_grpo_online():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_grpo_online"))
    dict_list_history = req.get("history", [])
    history = []
    for turn in dict_list_history:
        choises = []
        for convesation_dict_list in turn:
            choises.append(cList.from_dicts(convesation_dict_list))
        history.append(choises)
    ref_model_server = req.get("ref_model_server")
    lr_init = req.get("lr_init", grpo_config.lr_init)
    lr_final = req.get("lr_final", grpo_config.lr_final)
    warmup_steps = req.get("warmup_steps", grpo_config.warmup_steps)
    train_batch_size = req.get("train_batch_size", grpo_config.train_batch_size)
    clip_eps = req.get("clip_eps", grpo_config.clip_eps)
    kl_weight = req.get("kl_weight", grpo_config.kl_weight)
    grad_cp_max_norm = req.get("grad_cp_max_norm", grpo_config.grad_cp_max_norm)
    accumulate_grad = req.get("accumulate_grad", grpo_config.accumulate_grad)
    n_train_each_episode = req.get(
        "n_train_each_episode", grpo_config.n_train_each_episode
    )
    save_weight_folder = req.get("save_weight_folder")
    save_weight_name = req.get("save_weight_name")
    begin_with_state_dir = req.get("begin_with_state_dir")
    return app.train_grpo_online(
        history=history,
        ref_model_server=ref_model_server,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        train_batch_size=train_batch_size,
        clip_eps=clip_eps,
        kl_weight=kl_weight,
        grad_cp_max_norm=grad_cp_max_norm,
        accumulate_grad=accumulate_grad,
        n_train_each_episode=n_train_each_episode,
        save_weight_folder=save_weight_folder,
        save_weight_name=save_weight_name,
        begin_with_state_dir=begin_with_state_dir,
    )


@route("/train_sft_online", method="POST")
def train_sft_online():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_sft_online"))
    dict_list_history = req.get("history", [])
    history = []
    for turn in dict_list_history:
        choises = []
        for convesation_dict_list in turn:
            choises.append(cList.from_dicts(convesation_dict_list))
        history.append(choises)
    lr_init = req.get("lr_init", grpo_config.lr_init)
    lr_final = req.get("lr_final", grpo_config.lr_final)
    warmup_steps = req.get("warmup_steps", grpo_config.warmup_steps)
    begin_with_state_dir = req.get("begin_with_state_dir")
    save_weight_folder = req.get("save_weight_folder")
    save_weight_name = req.get("save_weight_name")
    ctx_len = req.get("ctx_len", train_config.model.ctx_len)
    return app.train_sft_online(
        history=history,
        lr_init=lr_init,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        begin_with_state_dir=begin_with_state_dir,
        save_weight_folder=save_weight_folder,
        save_weight_name=save_weight_name,
        ctx_len=ctx_len,
    )


async def distribute_package(received_package, route):
    async def send_to_rank(r):
        other_rank_port = global_config.server_config.train.port_begin + r
        url = f"http://localhost:{other_rank_port}/{route}"
        try:
            print(f"distribute package from {rank} to rank {r}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=received_package) as response:
                    await response.text()
        except Exception as e:
            print(f"Error sending package to rank {r}: {e}")

    tasks = []
    for r in range(total_ranks): 
        if r != rank:  # 不需要将数据包发送到自己
            tasks.append(send_to_rank(r))

    await asyncio.gather(*tasks)


run(host="0.0.0.0", port=port)
