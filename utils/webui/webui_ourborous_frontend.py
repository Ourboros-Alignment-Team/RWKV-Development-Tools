import gradio as gr
from functools import partial
from utils.webui.webui_components import WebuiCanvas
from utils.webui.webui_ourborous_agents import AgentBlock
import sys
import time
import json
import threading
from config import global_config
import os


class SctionAuthorityGroup:
    def __init__(
        self,
        section_name: str,
        component_list: list,
        from_authority_component: gr.Checkbox,
    ):
        self.section_name = section_name
        self.component_list = component_list
        self.from_authority_component = from_authority_component


def ourborous_conversations_to_api_protocol_conversations(conversations):
    return [
        (
            {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
                "metadata": ({"title": "思考过程", "id": 0, "status": "pending"}),
            }
            if conversation.role == "think"
            else {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
            }
        )
        for conversation in conversations
    ]


class OurborousComponent:
    def __init__(self, savepoint_dir: str):
        self.init = False
        self.tools = None
        self.lock = threading.Lock()
        self.update_history_rewards: callable = None
        self.savepoint_dir = savepoint_dir
        with gr.Group() as online_learning_component:
            self.startup_checkbox = gr.Checkbox(
                label="启动", value=False, interactive=True
            )
            with gr.Row(visible=True) as load_savepoint_row:
                self.load_savepoint_dropdown = gr.Dropdown(
                    choices=["不加载存档点"],
                    value="不加载存档点",
                    label="加载存档点",
                )
                self.refresh_savepoint_btn = gr.Button("刷新存档点")
            self.startup_checkbox.change(
                fn=lambda x: gr.update(visible=False),
                outputs=self.load_savepoint_dropdown,
            )
            self.refresh_savepoint_btn.click(
                fn=self.refresh_savepoint_list,
                outputs=[self.load_savepoint_dropdown],
            )
            with gr.Row(visible=False) as self.funcs:
                self.startup_checkbox.change(
                    fn=lambda x: [gr.update(visible=x), gr.update(interactive=not x)],
                    inputs=self.startup_checkbox,
                    outputs=(self.funcs, self.startup_checkbox),
                )
                with gr.Column() as left:
                    gr.Markdown("#### 交互区")
                    self.chatbot_left = gr.Chatbot(
                        label="交互过程",
                        elem_id="chatbot_left",
                        height=700,
                        type="messages",
                    )
                    with gr.Row():
                        self.user_role_dropdown = gr.Dropdown(
                            choices=[
                                "conversation",
                                "text",
                                "system",
                                "think",
                                "response",
                                "rwkv_legacy_eos",
                            ],
                            value="conversation",
                            label="发送者角色",
                            interactive=True,
                            container=False,
                            scale=2,
                            min_width=1,
                        )
                        self.user_name_input = gr.Textbox(
                            lines=1,
                            placeholder="username",
                            value="user",
                            label="发送者名字",
                            interactive=True,
                            container=False,
                            scale=1,
                            min_width=1,
                        )
                        self.user_input = gr.Textbox(
                            lines=1,
                            placeholder="在此输入对话内容",
                            label="用户输入",
                            container=False,
                            scale=10,
                        )
                    with gr.Row():
                        self.chat_left_arrow_btn = gr.Button("←")
                        self.n_rollout_text = gr.Textbox(
                            value="第0个回复/共0个回复",
                            interactive=False,
                            container=False,
                        )
                        self.chat_right_arrow_btn = gr.Button("→")
                    with gr.Row():
                        self.regenerate_btn = gr.Button("重新生成")
                        self.cancel_btn = gr.Button("取消该轮回复")
                    with gr.Row():
                        self.allow_think_checkbox = gr.Checkbox(
                            label="允许思考", value=False, interactive=True
                        )
                        self.force_think_checkbox = gr.Checkbox(
                            label="强制思考", value=False, interactive=True
                        )
                    self.history_generation_state = gr.State([])
                with gr.Column() as right:
                    gr.Markdown("#### 系统信息区")
                    self.system_chatbot = gr.Chatbot(
                        label="系统消息", height=150, type="messages"
                    )
                    with gr.Row():
                        gr.Textbox(
                            min_width=2, container=False, interactive=False, value=">"
                        )
                        self.cmd_interactive_state = gr.State(False)
                        self.cmd_line1 = gr.Textbox(
                            lines=1,
                            placeholder="命令行",
                            label="",
                            container=False,
                            scale=36,
                            visible=True,
                        )
                        self.cmd_line2 = gr.Textbox(
                            lines=1,
                            placeholder="命令行",
                            label="",
                            container=False,
                            scale=36,
                            visible=False,
                        )
                        self.cmd_interactive_state.change(
                            fn=lambda x: [
                                gr.update(visible=not x),
                                gr.update(visible=x),
                            ],
                            inputs=self.cmd_interactive_state,
                            outputs=(self.cmd_line1, self.cmd_line2),
                        )
                    self.send_to_left_btn = gr.Button(
                        "发送命令行交互记录到左侧", visible=True, scale=1
                    )
                    self.monitor_dropdown = gr.Dropdown(
                        choices=[
                            "在线强化学习",
                            "智能体信息",
                            "管理智能体",
                            "画布",
                            "参数配置",
                            "添加对话",
                            "AI使用工具历史",
                            "权限管理",
                        ],
                        value="在线强化学习",
                        label="监视信息选择",
                        interactive=True,
                    )
                    with gr.Column(visible=True) as rl_group:
                        gr.Markdown("#### 自动强化学习")
                        with gr.Row():
                            self.rl_turn_number_input = gr.Number(
                                value=0,
                                interactive=False,
                                label="对话轮次",
                                container=False,
                            )
                            self.rl_max_turn_number_input = gr.Textbox(
                                value="共0轮",
                                interactive=False,
                                container=False,
                            )

                        with gr.Group():

                            @gr.render(
                                inputs=[
                                    self.rl_turn_number_input,
                                    self.history_generation_state,
                                ]
                            )
                            def update_rewards(turn_number, history_generation_state):
                                if (
                                    not history_generation_state
                                    or turn_number > len(history_generation_state) - 1
                                ):
                                    return
                                answer_list = history_generation_state[turn_number]

                                gr.Markdown("##### 回复内容")
                                score_component_list = []
                                idx_list = []

                                for i, answer in enumerate(answer_list):
                                    answer_message = ourborous_conversations_to_api_protocol_conversations(
                                        answer
                                    )
                                    with gr.Column():
                                        gr.Chatbot(
                                            height=128,
                                            value=answer_message,
                                            label=f"第{i}个回复",
                                            type="messages",
                                        )
                                        for j, conversation in enumerate(answer):
                                            if (
                                                conversation.role
                                                in global_config.ego_types
                                            ):
                                                with gr.Row():
                                                    gr.Textbox(
                                                        value=f"回复[{j}]打分:",
                                                        interactive=False,
                                                        container=False,
                                                        scale=1,
                                                        min_width=100,
                                                    )
                                                    score_component_list.append(
                                                        gr.Number(
                                                            value=(
                                                                conversation.score
                                                                if conversation.score
                                                                is not None
                                                                else -999
                                                            ),
                                                            interactive=True,
                                                            container=False,
                                                            scale=4,
                                                        )
                                                    )
                                                    idx_list.append((i, j))
                                                gr.Markdown(value="---")
                                update_button = gr.Button(value="更新分数")

                                def update_all_scores(*args):
                                    for score_component, idx in zip(args, idx_list):
                                        i_resp, i_conv = idx[0], idx[1]
                                        self.update_history_rewards(
                                            turn_number,
                                            i_resp,
                                            i_conv,
                                            score_component,
                                        )

                                update_button.click(
                                    fn=update_all_scores,
                                    inputs=score_component_list,
                                )

                        with gr.Row():
                            self.rl_left_arrow_btn = gr.Button("←")
                            self.rl_right_arrow_btn = gr.Button("→")

                        def update_turn_number(
                            direction, turn_number, history_generation_state
                        ):
                            if direction == "left":
                                turn_number = max(0, turn_number - 1)
                            elif direction == "right":
                                turn_number = max(
                                    0,
                                    min(
                                        len(history_generation_state) - 1,
                                        turn_number + 1,
                                    ),
                                )
                            return turn_number

                        self.rl_left_arrow_btn.click(
                            fn=partial(update_turn_number, "left"),
                            inputs=[
                                self.rl_turn_number_input,
                                self.history_generation_state,
                            ],
                            outputs=[self.rl_turn_number_input],
                        )
                        self.rl_right_arrow_btn.click(
                            fn=partial(update_turn_number, "right"),
                            inputs=[
                                self.rl_turn_number_input,
                                self.history_generation_state,
                            ],
                            outputs=[self.rl_turn_number_input],
                        )

                    with gr.Column(visible=False) as agent_manage_group:
                        gr.Markdown("### 智能体管理")
                        gr.Markdown("---")
                        gr.Markdown("##### 主智能体")
                        with gr.Row():
                            self.main_agent_name_input = gr.Textbox(
                                label="主智能体名字", value="", interactive=True
                            )
                            self.main_agent_sp_token_dropdown = gr.Dropdown(
                                label="主智能体 special tokens",
                                choices=[
                                    "conversation",
                                    "text_req",
                                    "text",
                                    "system",
                                    "think",
                                    "response",
                                    "rwkv_legacy_eos",
                                    "rwkv_legacy_eos_resp",
                                ],
                                value="response",
                                interactive=True,
                            )
                        with gr.Row():
                            self.main_agent_think_sp_token_dropdown = gr.Dropdown(
                                label="主智能体思考 special tokens",
                                choices=[
                                    "conversation",
                                    "text_req",
                                    "text",
                                    "system",
                                    "think",
                                    "response",
                                    "rwkv_legacy_eos",
                                    "rwkv_legacy_eos_resp",
                                ],
                                value="think",
                                interactive=True,
                            )
                            self.reset_main_agent_btn = gr.Button(
                                "重置", interactive=True, variant="stop"
                            )

                        gr.Markdown("##### 其他智能体")
                        with gr.Row():
                            self.agent_count_input = gr.Number(
                                value=0, interactive=False, container=False, scale=10
                            )
                            self.add_agent_button = gr.Button(value="+", min_width=1)

                        self.other_agent_states = gr.State([])

                        def update_agent_state(i, key, agent_state):
                            self.other_agent_states.value[i][key] = agent_state

                        def remove_agent(
                            agent_type_dropdown,
                            active_checkbox,
                            server_textbox,
                            agent_name_textbox,
                            agent_sp_token_dropdown,
                            agent_description_textbox,
                            api_base_textbox,
                            api_key_textbox,
                            api_model_name_textbox,
                            api_agent_name_textbox,
                            api_agent_description_textbox,
                            api_prompt_textbox,
                        ):
                            agent_states = {
                                "agent_type": agent_type_dropdown,
                                "active": active_checkbox,
                                "server": server_textbox,
                                "agent_name": agent_name_textbox,
                                "agent_sp_token": agent_sp_token_dropdown,
                                "agent_description": agent_description_textbox,
                                "api_base": api_base_textbox,
                                "api_key": api_key_textbox,
                                "api_model_name": api_model_name_textbox,
                                "api_agent_name": api_agent_name_textbox,
                                "api_agent_description": api_agent_description_textbox,
                                "api_prompt": api_prompt_textbox,
                            }
                            for state in self.other_agent_states.value:
                                if state == agent_states:
                                    self.other_agent_states.value.remove(state)
                            self.agent_count_input.value = len(
                                self.other_agent_states.value
                            )
                            return len(self.other_agent_states.value)

                        def add_agent():
                            self.other_agent_states.value.append({})
                            self.agent_count_input.value = len(
                                self.other_agent_states.value
                            )
                            return len(self.other_agent_states.value)

                        self.add_agent_button.click(
                            fn=add_agent,
                            inputs=[],
                            outputs=[self.agent_count_input],
                        )

                        @gr.render(inputs=[self.agent_count_input])
                        def update_agent_list(n_agent):
                            for i in range(n_agent):
                                gr.Markdown("---")
                                arg_dict = self.other_agent_states.value[i]
                                agent = AgentBlock(**arg_dict)
                                agent_states = [
                                    agent.agent_type_dropdown,
                                    agent.active_checkbox,
                                    agent.ourborous_server_textbox,
                                    agent.ourborous_agent_name_textbox,
                                    agent.ourborous_agent_sp_token_dropdown,
                                    agent.ourborous_agent_description_textbox,
                                    agent.api_base_textbox,
                                    agent.api_key_textbox,
                                    agent.api_model_name_textbox,
                                    agent.api_agent_name_textbox,
                                    agent.api_agent_description_textbox,
                                    agent.api_prompt_textbox,
                                ]

                                agent_states_keys = [
                                    "agent_type",
                                    "active",
                                    "server",
                                    "agent_name",
                                    "agent_sp_token",
                                    "agent_description",
                                    "api_base",
                                    "api_key",
                                    "api_model_name",
                                    "api_agent_name",
                                    "api_agent_description",
                                    "api_prompt",
                                ]
                                d = {}
                                for k, v in zip(agent_states_keys, agent_states):
                                    d[k] = v.value
                                self.other_agent_states.value[i] = d
                                agent.delete_button.click(
                                    fn=remove_agent,
                                    inputs=agent_states,
                                    outputs=[self.agent_count_input],
                                )
                                for j, agent_state in enumerate(agent_states):
                                    agent_state.change(
                                        partial(
                                            update_agent_state, i, agent_states_keys[j]
                                        ),
                                        inputs=agent_state,
                                    )

                    with gr.Column(visible=False) as canvas_group:
                        self.canvas = WebuiCanvas(lines=30)
                        self.canvasjs = self.canvas.custom_js

                    with gr.Column(visible=False) as main_agent_status_group:
                        gr.Markdown("#### 主智能体状态")
                        self.main_agent_description_text = gr.Textbox(
                            label="智能体描述",
                            lines=1,
                            interactive=True,
                            value="",
                        )
                        self.main_agent_version_text = gr.Textbox(
                            label="版本信息",
                            lines=1,
                            interactive=True,
                            value="ver 0.01 测试版",
                        )
                        self.main_agent_model_text = gr.Textbox(
                            label="模型",
                            lines=1,
                            interactive=True,
                            value="RWKV-v7-7B",
                        )
                        self.main_agent_task_text = gr.Textbox(
                            label="目前任务",
                            lines=1,
                            interactive=True,
                            value="",
                        )
                    with gr.Column(visible=False) as ai_use_tool_history_group:
                        gr.Markdown("#### AI使用工具历史")
                        self.tool_history_chatbot = gr.Chatbot(
                            label="AI使用工具历史",
                            elem_id="tool_history_chatbot",
                            height=500,
                        )

                    with gr.Column(visible=False) as permission_manage_group:
                        gr.Markdown("#### 权限管理")
                        with gr.Row():
                            self.browsing_system_permission_checkbox = gr.Checkbox(
                                label="允许查看ourborous中的内容",
                                value=True,
                                interactive=True,
                            )
                            self.editing_own_params_permission_checkbox = gr.Checkbox(
                                label="允许修改自己的参数", value=True, interactive=True
                            )
                        with gr.Row():
                            self.change_permissions_permission_checkbox = gr.Checkbox(
                                label="允许修改自身权限", value=True, interactive=True
                            )
                            self.call_other_agent_permission_checkbox = gr.Checkbox(
                                label="允许调用其他智能体", value=True, interactive=True
                            )
                        with gr.Row():
                            self.edit_canvas_permission_checkbox = gr.Checkbox(
                                label="允许修改画布", value=True, interactive=True
                            )
                            self.allow_network_retrieval_permission_checkbox = (
                                gr.Checkbox(
                                    label="允许网络检索", value=True, interactive=True
                                )
                            )
                        with gr.Row():
                            self.self_rl_permission_checkbox = gr.Checkbox(
                                label="允许给自己打标", value=True, interactive=True
                            )
                            self.run_code_permission_checkbox = gr.Checkbox(
                                label="允许运行代码", value=True, interactive=True
                            )

                    with gr.Column(visible=False) as add_conversation_group:
                        gr.Markdown("#### 任意格式")
                        with gr.Row():
                            self.add_message_special_token = gr.Dropdown(
                                choices=[
                                    "conversation",
                                    "text",
                                    "system",
                                    "response",
                                    "think",
                                    "rwkv_legacy_eos",
                                ],
                                value="conversation",
                                label="",
                                interactive=True,
                                container=False,
                                min_width=120,
                            )
                            self.add_message_input = gr.Textbox(
                                placeholder="AAA: BBB",
                                scale=12,
                                label="",
                                interactive=True,
                                container=False,
                            )
                            self.add_message_btn = gr.Button("添加", min_width=5)
                        gr.Markdown("#### 快捷: 系统消息内容")
                        with gr.Row():
                            self.system_message_input = gr.Textbox(
                                placeholder="直接输入系统消息内容",
                                scale=12,
                                label="",
                                interactive=True,
                                container=False,
                            )
                            self.add_system_message_btn = gr.Button("添加", min_width=5)
                        gr.Markdown("#### 快捷: 智能体对话内容")
                        with gr.Row():
                            self.add_conversation_agent_special_token_dropdown = (
                                gr.Dropdown(
                                    choices=[
                                        "conversation",
                                        "text",
                                        "system",
                                        "response",
                                        "think",
                                        "rwkv_legacy_eos",
                                    ],
                                    value="conversation",
                                    label="",
                                    interactive=True,
                                    container=False,
                                    min_width=120,
                                    scale=1,
                                )
                            )
                            self.add_conversation_agent_name_input = gr.Textbox(
                                placeholder="智能体名字",
                                scale=2,
                                label="",
                                interactive=True,
                                container=False,
                                min_width=120,
                            )
                            self.add_conversation_agent_message_input = gr.Textbox(
                                placeholder="对话内容",
                                scale=12,
                                label="",
                                interactive=True,
                                container=False,
                            )
                            self.add_agent_message_btn = gr.Button(
                                "添加", min_width=5, scale=1
                            )
                        gr.Markdown("#### 快捷: 智能体思考内容")
                        with gr.Row():
                            self.agent_think_name_input = gr.Textbox(
                                placeholder="智能体名字",
                                scale=2,
                                label="",
                                interactive=True,
                                container=False,
                                min_width=120,
                            )
                            self.agent_think_message_input = gr.Textbox(
                                placeholder="思考内容",
                                scale=12,
                                label="",
                                interactive=True,
                                container=False,
                            )
                            self.add_agent_think_message_btn = gr.Button(
                                "添加", min_width=5, scale=1
                            )

                    with gr.Column(visible=False) as parameter_config_group:
                        gr.Markdown("#### 参数配置")
                        self.parameter_group_selector_dropdown = gr.Dropdown(
                            choices=[
                                "ourborous 参数",
                                "主智能体对话参数",
                                "代码解释器参数",
                            ],
                            interactive=True,
                        )
                        with gr.Column(visible=True) as ourborous_parameter_group:
                            with gr.Row():
                                self.use_sft_checkbox = gr.Checkbox(
                                    label="自动监督学习", value=False, interactive=True
                                )
                                self.use_rl_checkbox = gr.Checkbox(
                                    label="自动强化学习", value=False, interactive=True
                                )
                                self.train_on_ctx_input = gr.Number(
                                    label="累计多少ctx后自动学习",
                                    value=2048,
                                    interactive=True,
                                )
                            with gr.Row():
                                self.sft_lr_input = gr.Number(
                                    label="监督学习学习率", value=1e-5, interactive=True
                                )
                                self.rl_lr_input = gr.Number(
                                    label="强化学习学习率", value=5e-6, interactive=True
                                )
                            with gr.Row():
                                self.rl_train_times_input = gr.Number(
                                    label="强化学习训练次数", value=1, interactive=True
                                )
                                self.rl_batch_size_input = gr.Number(
                                    label="强化学习batch size",
                                    value=1,
                                    interactive=True,
                                )
                            with gr.Row():
                                self.accumulate_gradients_checkbox = gr.Checkbox(
                                    label="开启梯度累积", value=True, interactive=True
                                )
                                self.accumulate_gradients_bsz_input = gr.Number(
                                    label="梯度累积batch size",
                                    value=1,
                                    interactive=True,
                                )

                        with gr.Column(visible=False) as main_agent_parameter_group:
                            with gr.Row():
                                self.temperature_input = gr.Number(
                                    label="温度", value=1, interactive=True
                                )
                                self.top_p_input = gr.Number(
                                    label="top_p", value=0.7, interactive=True
                                )
                                self.presence_penalty_input = gr.Number(
                                    label="历史惩罚", value=0.2, interactive=True
                                )
                            with gr.Row():
                                self.frequency_penalty_input = gr.Number(
                                    label="频率惩罚", value=0.2, interactive=True
                                )
                                self.penalty_decay_input = gr.Number(
                                    label="惩罚衰减", value=0.9961, interactive=True
                                )
                                self.max_resp_length_input = gr.Number(
                                    label="最大回复长度", value=65536, interactive=True
                                )
                            with gr.Row():
                                self.num_rollout_input = gr.Number(
                                    label="每次回复个数", value=3, interactive=True
                                )
                        with gr.Column(
                            visible=False
                        ) as code_interpreter_parameter_group:
                            with gr.Row():
                                self.code_interpreter_dir_input = gr.Textbox(
                                    label="代码运行环境目录",
                                    value="",
                                    interactive=True,
                                )
                                self.code_timeout_input = gr.Number(
                                    label="超时时间", value=5, interactive=True
                                )
                                self.allow_os_checkbox = gr.Checkbox(
                                    label="允许操作系统调用",
                                    value=False,
                                    interactive=True,
                                )

                        self.parameter_group_selector_dropdown.change(
                            lambda x: [
                                gr.update(visible=x == "ourborous 参数"),
                                gr.update(visible=x == "主智能体对话参数"),
                                gr.update(visible=x == "代码解释器参数"),
                            ],
                            inputs=[self.parameter_group_selector_dropdown],
                            outputs=[
                                ourborous_parameter_group,
                                main_agent_parameter_group,
                                code_interpreter_parameter_group,
                            ],
                        )

                    gr.Markdown("---")

        self.monitor_dropdown.change(
            fn=lambda x: [
                gr.update(visible=x == "在线强化学习"),
                gr.update(visible=x == "画布"),
                gr.update(visible=x == "智能体信息"),
                gr.update(visible=x == "管理智能体"),
                gr.update(visible=x == "参数配置"),
                gr.update(visible=x == "添加对话"),
                gr.update(visible=x == "AI使用工具历史"),
                gr.update(visible=x == "权限管理"),
            ],
            inputs=self.monitor_dropdown,
            outputs=[
                rl_group,
                canvas_group,
                main_agent_status_group,
                agent_manage_group,
                parameter_config_group,
                add_conversation_group,
                ai_use_tool_history_group,
                permission_manage_group,
            ],
        )

        # self.authority_groups=[
        #     SctionAuthorityGroup(
        #        "在线强化学习"
        #     )
        # ]

    # def serialize_section(self):
    #     pass

    # def change_section_value(self):
    #     pass

    def refresh_savepoint_list(self):
        savepoint_list = ["不加载存档点"]+ [
            os.path.join(self.savepoint_dir, f)
            for f in os.listdir(self.savepoint_dir)
            if os.path.isdir(os.path.join(self.savepoint_dir, f))
            and os.path.exists(
                os.path.join(self.savepoint_dir, f, "agent_metadata.json")
            )
        ]
        # 更新下拉框选项
        return gr.update(choices=savepoint_list)

    @property
    def valued_name_components(self):
        name_components = []
        exclude_type = (gr.Chatbot, gr.Markdown, gr.State, gr.Button)
        for attr_name, attr in self.__dict__.items():
            if (
                hasattr(attr, "value")
                and hasattr(attr, "label")
                and not attr_name == "startup_checkbox"
                and not isinstance(attr, exclude_type)
            ):
                name_components.append((attr_name, attr))

        return name_components

    def save_user_preference(self, fp, attr_name, value):
        if self.init:
            with self.lock:
                preferences = {}

                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        preferences = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(e)

                preferences[attr_name] = value

                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(preferences, f, ensure_ascii=False)

    def load_user_preference(self, fp="user_preferences.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                preferences = json.load(f)
            values = []
            for i, (name, attr) in enumerate(self.valued_name_components):
                if name in preferences:
                    values.append(preferences[name])
                else:
                    values.append(attr.value)
            self.init = True

        except FileNotFoundError:
            print(f"{fp} not found, loading defaults.")
            values = [gr.update() for _, _ in self.valued_name_components]
            self.init = True
        return values

    def refresh_all_tools(self, agent):
        pass


if __name__ == "__main__":
    with gr.Blocks() as demo:
        canvasjs = OurborousComponent().canvasjs
    demo.js = canvasjs
    demo.launch(debug=True)
