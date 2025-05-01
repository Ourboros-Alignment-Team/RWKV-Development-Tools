from utils.ourborous.agent import OurborousAgent
from utils.message_manager import Conversation, cList
import gradio as gr
from config import global_config
from threading import Lock
import time


def ourborous_chat(
    agent: OurborousAgent,
    regenerate: bool,
    sender_role: str,
    sender_name: str,
    sender_text: str,
    rpy_role: str,
    replier_name: str,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    alpha_frequency: float,
    alpha_presence: float,
    alpha_decay: float = 0.9961,
    max_resp_len: float = 2048,
    allow_think: bool = False,
    force_think: bool = False,
):
    send_convcersations = cList.from_single_conversation(
        Conversation(
            role=sender_role,
            content=f"{sender_name}: {sender_text}" if sender_name else sender_text,
        )
    )
    for xxx in agent.chat(
        regenerate=regenerate,
        send_conversations=send_convcersations,
        rpy_role=rpy_role,
        replier_name=replier_name,
        num_rollouts=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        alpha_frequency=alpha_frequency,
        alpha_presence=alpha_presence,
        alpha_decay=alpha_decay,
        max_resp_len=max_resp_len,
        format_constraint_str=None,
        token_ban=[0],
        allow_think=allow_think,
        force_think=force_think,
    ):
        n_resps = (
            len(agent.history_container.untrained_histories[-1])
            if agent.history_container.untrained_histories
            else 0
        )
        lookat_and_ctx_str = f"第{agent.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {agent.untrained_ctx_len})"
        yield xxx, lookat_and_ctx_str, False
    yield xxx, lookat_and_ctx_str, True


def ourborous_regenerate(
    agent: OurborousAgent,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    alpha_frequency: float,
    alpha_presence: float,
    alpha_decay: float = 0.9961,
    max_resp_len: float = 2048,
    allow_think: bool = False,
    force_think: bool = False,
):
    for xxx in agent.regenerate(
        num_rollouts=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        alpha_frequency=alpha_frequency,
        alpha_presence=alpha_presence,
        alpha_decay=alpha_decay,
        max_resp_len=max_resp_len,
        format_constraint_str=None,
        token_ban=[0],
        allow_think=allow_think,
        force_think=force_think,
    ):
        n_resps = (
            len(agent.history_container.untrained_histories[-1])
            if agent.history_container.untrained_histories
            else 0
        )
        lookat_and_ctx_str = f"第{agent.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {agent.untrained_ctx_len})"
        yield xxx, lookat_and_ctx_str, False
    yield xxx, lookat_and_ctx_str, True

def ourborous_back_to_last(agent):
    agent.back_to_last()
    api_protocol_conversations = (
        agent.ourborous_conversations_to_api_protocol_conversations(
            agent.gather_hist_conversation(
                agent.history_container.trained_histories
                + agent.history_container.untrained_histories,
                last_idx=agent.lookat_idx,
            )
        )
    )
    return api_protocol_conversations,True

def ourborous_load_savepoint(agent, path_cmd: str):
    if path_cmd == "不加载存档点":
        return gr.update(), gr.update()
    else:
        agent.load_savepoint(path_cmd)
        return refresh_ourborous(agent)

def refresh_ourborous(agent):
    api_protocol_conversations = (
        agent.ourborous_conversations_to_api_protocol_conversations(
            agent.gather_hist_conversation(
                agent.history_container.trained_histories
                + agent.history_container.untrained_histories,
                last_idx=agent.lookat_idx,
            )
        )
    )
    n_resps = (
        len(agent.history_container.untrained_histories[-1])
        if agent.history_container.untrained_histories
        else 0
    )
    lookat_and_ctx_str = f"第{agent.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {agent.untrained_ctx_len})"
    return api_protocol_conversations, lookat_and_ctx_str


def ourborous_refresh_rl_history(agent):
    history = []
    for turn in agent.history_container.untrained_histories:
        turn_choices = []
        for conversations in turn:
            conversations: cList
            new_conversations = cList()
            for i, conversation in enumerate(conversations):
                new_conversation = Conversation(
                    role=conversation.role,
                    content=conversation.content,
                    score=conversation.score,
                )
                if new_conversation.role in global_config.ego_types:
                    new_conversation.content = f"(回复[{i}]) {new_conversation.content}"
                new_conversations.append(new_conversation)
            turn_choices.append(new_conversations)
        history.append(turn_choices)
    n_turns = len(history)
    return history, n_turns


def ourborous_update_history_rewards(
    agent, turn: int, resp_idx: int, conversation_idx: int, rewards: float
):
    time.sleep(0.1)
    if rewards != -999:
        print(
            f"更新第{turn}轮第{resp_idx}个回复第{conversation_idx}个对话{agent.history_container.untrained_histories[turn][resp_idx][conversation_idx]}的reward为{rewards}"
        )
        agent.history_container.untrained_histories[turn][resp_idx][
            conversation_idx
        ].score = rewards


def reset_ourborous(agent):
    agent.reset()
    return [], 0


def refresh_all_tools(agent):
    # 自动强化学习
    history, n_turns = ourborous_refresh_rl_history(agent)
    tools = agent.tools
    if agent.temp_tools:
        tools = agent.temp_tools[agent.lookat_idx]
    # 画布
    canvas_text = tools.canvas.txt
    # 推理参数
    temperature = tools.params.temperature
    top_p = tools.params.top_p
    presence_penalty = tools.params.presence_penalty
    frequency_penalty = tools.params.frequency_penalty
    panalty_decay = tools.params.penalty_decay
    max_resp_len = tools.params.max_resp_length
    num_rollouts = tools.params.num_rollouts
    # 权限
    browsing_system_permission = tools.permissions.browsing_system_permission
    editing_params_permission = tools.permissions.editing_params_permission
    change_permissions_permission = tools.permissions.change_permissions_permission
    canvas_permission = tools.permissions.canvas_permission
    retrieving_permission = tools.permissions.retrieving_permission
    rl_permission = tools.permissions.rl_permission
    run_code_permission = tools.permissions.run_code_permission
    at_agents_permission = tools.permissions.at_agents_permission
    history, n_turns = ourborous_refresh_rl_history(agent)

    # for turn in history:
    #     for resp in turn:
    #         for conv in resp:
    #             print(conv, "==>", conv.score)

    return (
        history,
        n_turns,
        canvas_text,
        temperature,
        top_p,
        presence_penalty,
        frequency_penalty,
        panalty_decay,
        max_resp_len,
        num_rollouts,
        browsing_system_permission,
        editing_params_permission,
        change_permissions_permission,
        canvas_permission,
        retrieving_permission,
        rl_permission,
        run_code_permission,
        at_agents_permission,
        False,
    )


def ourborous_change_ollr_param(
    agent,
    use_sft,
    use_rl,
    train_on_ctx,
    sft_lr,
    rl_lr,
    rl_train_times,
    rl_batch_size,
    accumulate_gradients,
    accumulate_gradients_bsz,
):
    agent.auto_train_on_ctx = train_on_ctx
    agent.auto_sft = use_sft
    agent.auto_rl = use_rl
    agent.trainner.auto_sft_lr = sft_lr
    agent.trainner.auto_rl_lr = rl_lr
    agent.trainner.auto_rl_times = rl_train_times
    agent.trainner.auto_rl_bsz = rl_batch_size
    agent.trainner.accumulate_grad = accumulate_gradients
    agent.trainner.accumulate_grad_bsz = accumulate_gradients_bsz


def ourborous_change_main_agent_params(
    agent,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    panalty_decay: float,
    max_resp_len: int,
    num_rollouts: int,
):
    if agent.temp_tools:
        agent.temp_tools[agent.lookat_idx].params.temperature = temperature
        agent.temp_tools[agent.lookat_idx].params.top_p = top_p
        agent.temp_tools[agent.lookat_idx].params.presence_penalty = presence_penalty
        agent.temp_tools[agent.lookat_idx].params.frequency_penalty = frequency_penalty
        agent.temp_tools[agent.lookat_idx].params.penalty_decay = panalty_decay
        agent.temp_tools[agent.lookat_idx].params.max_resp_len = max_resp_len
        agent.temp_tools[agent.lookat_idx].params.num_rollouts = num_rollouts
    else:
        agent.tools.params.temperature = temperature
        agent.tools.params.top_p = top_p
        agent.tools.params.presence_penalty = presence_penalty
        agent.tools.params.frequency_penalty = frequency_penalty
        agent.tools.params.penalty_decay = panalty_decay
        agent.tools.params.max_resp_len = max_resp_len
        agent.tools.params.num_rollouts = num_rollouts


def ourborous_change_main_agent_status(
    agent,
    name,
    description,
    version,
    model_type,
    now_focus,
):
    if agent.temp_tools:
        agent.temp_tools[agent.lookat_idx].status.name = name
        agent.temp_tools[agent.lookat_idx].status.description = description
        agent.temp_tools[agent.lookat_idx].status.version = version
        agent.temp_tools[agent.lookat_idx].status.model_type = model_type
        agent.temp_tools[agent.lookat_idx].status.now_focus = now_focus
    else:
        agent.tools.status.name = name
        agent.tools.status.description = description
        agent.tools.status.version = version
        agent.tools.status.model_type = model_type
        agent.tools.status.now_focus = now_focus


def ourborous_change_permissions(
    agent,
    browsing_system_permission,
    editing_params_permission,
    change_permissions_permission,
    canvas_permission,
    retrieving_permission,
    rl_permission,
    run_code_permission,
    at_agents_permission,
):
    if agent.temp_tools:
        agent.temp_tools[
            agent.lookat_idx
        ].permissions.browsing_system_permission = browsing_system_permission
        agent.temp_tools[
            agent.lookat_idx
        ].permissions.editing_params_permission = editing_params_permission
        agent.temp_tools[
            agent.lookat_idx
        ].permissions.change_permissions_permission = (
            change_permissions_permission
        )
        agent.temp_tools[agent.lookat_idx].permissions.canvas_permission = (
            canvas_permission
        )
        agent.temp_tools[agent.lookat_idx].permissions.retrieving_permission = (
            retrieving_permission
        )
        agent.temp_tools[agent.lookat_idx].permissions.rl_permission = (
            rl_permission
        )
        agent.temp_tools[agent.lookat_idx].permissions.run_code_permission = (
            run_code_permission
        )
        agent.temp_tools[agent.lookat_idx].permissions.at_agents_permission = (
            at_agents_permission
        )
    agent.tools.permissions.browsing_system_permission = browsing_system_permission
    agent.tools.permissions.editing_params_permission = editing_params_permission
    agent.tools.permissions.change_permissions_permission = (
        change_permissions_permission
    )
    agent.tools.permissions.canvas_permission = canvas_permission
    agent.tools.permissions.retrieving_permission = retrieving_permission
    agent.tools.permissions.rl_permission = rl_permission
    agent.tools.permissions.run_code_permission = run_code_permission
    agent.tools.permissions.at_agents_permission = at_agents_permission


def ourborous_change_canvas(
    agent,
    canvas_text,
):
    if agent.temp_tools:
        agent.temp_tools[agent.lookat_idx].canvas.txt = canvas_text
    else:
        agent.tools.canvas.txt = canvas_text


def user_agent_cmd(user_agent, msg, sender_role, sender_name):
    api_protocol_conversations = None
    for api_protocol_conversations in user_agent.receive_message(
        msg, sender_role, sender_name
    ):
        yield api_protocol_conversations, True, True, gr.update(
            visible=bool(user_agent.act_history)
        )
    if api_protocol_conversations is not None:
        yield api_protocol_conversations, False, True, gr.update(
            visible=bool(user_agent.act_history)
        )


def send_usr_agent_cmd_hist_to_main_agent(
    agent,
    user_agent,
    rpy_role: str,
    replier_name: str,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    alpha_frequency: float,
    alpha_presence: float,
    alpha_decay: float = 0.9961,
    max_resp_len: float = 2048,
    allow_think: bool = False,
    force_think: bool = False,
):
    send_convcersations = user_agent.act_history

    for xxx in agent.chat(
        regenerate=False,
        send_conversations=send_convcersations,
        rpy_role=rpy_role,
        replier_name=replier_name,
        num_rollouts=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        alpha_frequency=alpha_frequency,
        alpha_presence=alpha_presence,
        alpha_decay=alpha_decay,
        max_resp_len=max_resp_len,
        format_constraint_str=None,
        token_ban=[0],
        allow_think=allow_think,
        force_think=force_think,
    ):
        n_resps = (
            len(agent.history_container.untrained_histories[-1])
            if agent.history_container.untrained_histories
            else 0
        )
        lookat_and_ctx_str = f"第{agent.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {agent.untrained_ctx_len})"
        yield xxx, lookat_and_ctx_str, False, gr.update(visible=bool(user_agent.act_history))
    user_agent.reset()
    yield xxx, lookat_and_ctx_str, True, gr.update(visible=bool(user_agent.act_history))


def ourborous_add_message(agent, sender_role, sender_name, msg):
    new_conversations = cList.from_single_conversation(
        Conversation(
            role=sender_role,
            content=f"{sender_name}: {msg}" if sender_name else msg,
        )
    )
    return agent.add_message(new_conversations)


def ourborous_add_think(agent, sender_name, msg):
    new_conversations = cList.from_single_conversation(
        Conversation(
            role="think",
            content=f"({sender_name}: {msg})" if sender_name else msg,
        )
    )
    return agent.add_message(new_conversations)


def ourborous_add_conversations(agent, sender_role, sender_content):
    new_conversations = cList.from_single_conversation(
        Conversation(
            role=sender_role,
            content=sender_content,
        )
    )
    return agent.add_message(new_conversations)


def ourborous_chat_lookat_pre(agent):
    api_protocol_conversations, lookat_and_ctx_str = agent.lookat_pre()
    return api_protocol_conversations, lookat_and_ctx_str, True


def ourborous_chat_lookat_next(agent):
    api_protocol_conversations, lookat_and_ctx_str = agent.lookat_next()
    return api_protocol_conversations, lookat_and_ctx_str, True
