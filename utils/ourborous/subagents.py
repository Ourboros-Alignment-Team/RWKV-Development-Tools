from utils.message_manager import cList, Conversation
from utils.llms_api_chatbot import OpenAPIChatBot
from utils.inference_chatbot import Chatbot
import gc
from utils.ourborous.chain_helper import ChainNode, CommandRouter, get_agent_response
from utils.message_manager import cList, Conversation
import re
from abc import ABC, abstractmethod


class OurborousSubAgents:
    def __init__(self):
        self.agents = []

    def get_router(self):
        rules = []
        for i, agent in enumerate(self.agents):
            if agent.activate:
                rules.append(
                    (
                        f"@{agent.agent_name} (.*)",
                        ChainNode(todo=self.act_chat_agent, next=None),
                    )
                )
        router = CommandRouter(
            rules=rules,
            default=ChainNode(todo=self.act_find_fault, next=None),
        )
        return router

    def add_agent(self, agent_type, **kwargs):
        if agent_type == "ourborous服务端模型":
            agent = OurborousAgentInterface(**kwargs)
        elif agent_type == "API模型":
            agent = PublicApiAgentInterface(**kwargs)
        self.agents.append(agent)

    def change_agent_type(self, index, agent_type, **kwargs):
        if agent_type == "ourborous服务端模型":
            agent = OurborousAgentInterface(**kwargs)
        elif agent_type == "API模型":
            agent = PublicApiAgentInterface(**kwargs)
        self.agents[index] = agent
        gc.collect()

    def remove_at_agent(self, index):
        self.agents.pop(index)

    def find(self, agent_name: str):
        for agent in self.agents:
            if agent.agent_name.strip() == agent_name.strip():
                return agent

    def act_find_fault(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        pattern = r"@(.*) (.*)"
        match_groups = re.match(pattern, detected_conversation.content, re.DOTALL)
        agent_name = match_groups.group(1)
        output = f"没有叫作{agent_name}的智能体。"
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_list_agents(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        output = ""
        if len(self.agents) == 0:
            output = "目前没有其他智能体可以调用。"
        for i, agent in enumerate(self.agents):
            output += f"""--- 智能体{i} ---
{str(agent)}"""
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_chat_agent(
        self,
        from_agent,
        detected_conversation: Conversation,
        act_history: cList,
        **kwargs,
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"@(.*) (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        agent_name = match_groups.group(1)
        query = match_groups.group(2)
        other_agent = self.find(agent_name)
        req_conversation = Conversation(role="conversation", content=query)
        out_conversation = other_agent.chat(req_conversation)
        act_history.append(out_conversation)
        for resp_conversation, act_history in get_agent_response(
            from_agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history


class AgentInterface(ABC):
    @abstractmethod
    def chat(self, messages: Conversation):
        pass

    @abstractmethod
    def __str__(self):
        pass


class OurborousAgentInterface(AgentInterface):
    def __init__(
        self,
        server_url: str,
        agent_name: str,
        agent_special_token_role: str,
        agent_description: str,
    ):
        self.activate = False
        self.server_url = server_url
        self.agent_name = agent_name
        self.agent_special_token_role = agent_special_token_role
        self.agent_description = agent_description
        self.inited = False
        self.chatbot = None

    def chat(self, messages: Conversation):
        if not self.inited:
            self.chatbot = Chatbot(
                server=self.server_url,
                bot_sp_token_role=self.agent_special_token_role,
                temp=None,
                top_p=None,
                presence_penalty=None,
                frequency_penalty=None,
                decay_penalty=None,
            )
        send_conversations = cList(messages)
        resp_full_str = self.chatbot.chat(
            send_conversations=send_conversations,
            rpy_prefix=f"{self.agent_name}:",
            stream=False,
        )
        return Conversation(
            role="conversation",
            content=f"{self.agent_name}: {resp_full_str}",
        )

    def reset(self):
        self.chatbot.reset()

    def __str__(self):
        return f"""名字: {self.agent_name}
{self.agent_description}
本地模型
使用`@{self.agent_name} <对话内容>`的格式即可与该智能体对话。
"""


class PublicApiAgentInterface(AgentInterface):
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        agent_model: str,
        agent_name: str,
        agent_description: str,
        prompt: str,
    ):
        self.activate = False
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.agent_model = agent_model
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.prompt = prompt

    def chat(self, messages: Conversation):
        chatbot = OpenAPIChatBot(
            api_base=self.api_base_url,
            api_key=self.api_key,
            use_model=self.agent_model,
        )
        try:
            send_msgs = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": messages.content},
            ]
            resp_full_str = chatbot.chat(send_msgs)
            return Conversation(
                role="conversation",
                content=f"{self.agent_name}: {resp_full_str}",
            )
        except Exception as e:
            resp_full_str = f"Error: {e}"
            return Conversation(
                role="system", content=f"智能体对话请求被中断，原因：{resp_full_str}"
            )

    def __str__(self):
        return f"""名字: {self.agent_name}
{self.agent_description}
API模型，模型名称：{self.agent_model}
"""
