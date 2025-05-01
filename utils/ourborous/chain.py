from utils.ourborous.chain_helper import ChainNode, CommandRouter, get_agent_response
from utils.ourborous.canvas import OurborousCanvas
from utils.ourborous.tools import (
    WikiRetriever,
    MainAgentStatus,
    PermissionManager,
    MainAgentParams,
)
from utils.ourborous.subagents import OurborousSubAgents
from utils.ourborous.code_runner import CodeRunner

# from utils.ourborous.rlfb import ReinforceLearningFeedback
from utils.message_manager import cList, Conversation
import re


class Ourboroustools:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        version: str = "",
        model_type: str = "",
        now_focus: str = "",
        browsing_system_permission: bool = True,
        editing_params_permission: bool = True,
        change_permissions_permission: bool = False,
        canvas_permission: bool = True,
        retrieving_permission: bool = True,
        rl_permission: bool = True,
        run_code_permission: bool = True,
        at_agents_permission: bool = True,
        temperature: float = 1,
        top_p: float = 0.8,
        presence_penalty: float = 0.2,
        frequency_penalty: float = 0.2,
        penalty_decay: float = 0.9961,
        max_resp_length: int = 65536,
        num_rollouts: int = 3,
        run_code_dir: str = ".",
        enable_os: bool = False,
        code_timeout: int = 3,
    ):
        self.help_text = """--- 指令一览 ---
* 帮助: /help
* 使用画布: /canvas
* 查看自身状态: /my_status
* 检索维基百科: /wiki_search <关键词>
* 查看和修改权限: /permissions
* 运行代码: /run_code ```python <代码>```
* 查看和修改参数: /params
* 查看其他智能体信息: /list_agents
* 向其他智能体发送消息: @<智能体名字> <消息>
--- ❤ ---
"""
        self.params = MainAgentParams(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            max_resp_length=max_resp_length,
            num_rollouts=num_rollouts,
        )
        self.code_runner = CodeRunner(
            run_code_dir=run_code_dir, enable_os=enable_os, timeout=code_timeout
        )
        self.permissions = PermissionManager(
            browsing_system_permission=browsing_system_permission,
            editing_params_permission=editing_params_permission,
            change_permissions_permission=change_permissions_permission,
            canvas_permission=canvas_permission,
            retrieving_permission=retrieving_permission,
            rl_permission=rl_permission,
            run_code_permission=run_code_permission,
            at_agents_permission=at_agents_permission,
        )
        self.canvas = OurborousCanvas()
        self.wiki_retriever = WikiRetriever()
        self.sub_agents = OurborousSubAgents()
        # self.rlfb = ReinforceLearningFeedback()
        self.status = MainAgentStatus(
            name=name,
            description=description,
            version=version,
            model_type=model_type,
            now_focus=now_focus,
        )
        self.router = CommandRouter(
            rules=None,
            default=None,
        )
        self.router_permissions_rules = [
            (
                r"/help",
                True,
                ChainNode(todo=self.act_help, next=self.router),
            ),
            (r"/canvas", self.permissions.canvas_permission, self.canvas.entry),
            (
                r"/my_status",
                self.permissions.browsing_system_permission,
                ChainNode(todo=self.status.act_get_status, next=None),
            ),
            (
                r"/list_agents",
                self.permissions.browsing_system_permission,
                ChainNode(todo=self.sub_agents.act_list_agents, next=None),
            ),
            (
                r"/wiki_search (.*)",
                self.permissions.retrieving_permission,
                ChainNode(todo=self.wiki_retriever.act_retrieval, next=None),
            ),
            (
                r"/permissions",
                self.permissions.change_permissions_permission,
                ChainNode(
                    todo=self.permissions.act_show_permissions,
                    next=ChainNode(
                        todo=self.permissions.act_change_permissions, next=None
                    ),
                ),
            ),
            (
                r"/run_code (.*)",
                self.permissions.run_code_permission,
                ChainNode(todo=self.code_runner.act_run_code, next=None),
            ),
            (
                r"/params",
                self.permissions.editing_params_permission,
                ChainNode(
                    todo=self.params.act_show_params,
                    next=ChainNode(todo=self.params.act_change_params, next=None),
                ),
            ),
            (
                r"@(.*) (.*)",
                self.permissions.at_agents_permission,
                self.sub_agents.get_router,
            ),
        ]
        rules = []
        for pattern, permission, node in self.router_permissions_rules:
            rules.append((pattern, node))

        self.entry = ChainNode(
            todo=self.act_detact_command,
            next=None,
        )
        self.router.rules = rules

    def act_detact_command(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        pattern_permissions = []
        for pattern, permission, todo in self.router_permissions_rules:
            pattern_permissions.append((pattern, permission))
        for i, (pattern, permission) in enumerate(pattern_permissions):
            match_groups = re.match(pattern, detected_conversation.content, re.DOTALL)
            if match_groups:
                if permission:
                    for resp_conversation, act_history in self.router.chain(
                        agent, detected_conversation, act_history, **kwargs
                    ):
                        yield resp_conversation, act_history
                    break
                else:
                    act_history.append(
                        Conversation(
                            role="system",
                            content=f"system: 你没有权限使用此指令，请向使用者申请权限。",
                        )
                    )
                    for resp_conversation, act_history in get_agent_response(
                        agent, act_history, **kwargs
                    ):
                        yield resp_conversation, act_history
                    break

    def act_help(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        act_history.append(
            Conversation(
                role="system",
                content=f"system: {self.help_text}",
            )
        )
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            print("chain resp: ", resp_conversation)
            yield resp_conversation, act_history
