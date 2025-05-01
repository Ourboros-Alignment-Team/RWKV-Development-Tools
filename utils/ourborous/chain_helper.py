from utils.message_manager import cList, Conversation
import re
from typing import Tuple, Union, List
from typing_extensions import Self

def remove_prefix(text, valid_prefixes):
    # 构建正则表达式，匹配特定的前缀
    pattern = rf'^({"|".join(map(re.escape, valid_prefixes))}):\s*'
    match = re.match(pattern, text, re.DOTALL)

    if match:
        # 如果匹配成功，删除匹配的部分并 strip
        return re.sub(pattern, "", text).strip()
    else:
        # 如果没有匹配，返回原字符串
        return text


def get_agent_response(
    agent,
    act_history: cList,
    **kwargs,
):
    if agent is not None:
        valid_prefix = kwargs.get("replier_name")
        if valid_prefix is None:
            valid_prefix = kwargs.get("sender_name")
        if valid_prefix is None:
            valid_prefix = ""

        replier_prefix = valid_prefix + ":" if valid_prefix != "" else ""
        for speak_full_text_batch in agent.generate(
            send_conversations=act_history,
            num_rollouts=1,
            **kwargs,
        ):
            resp_conversation = Conversation(
                role=kwargs["rpy_role"],
                content=f"{replier_prefix} {speak_full_text_batch[0]}",
            )
            temp_hist = act_history + cList([resp_conversation])
            yield resp_conversation, temp_hist
        act_history.append(resp_conversation)
        yield resp_conversation, act_history


class ChainLogger:
    def __init__(self):
        self.history = []

    def clear(self):
        self.history.clear()

    def log(self, history: cList):
        self.history.append(history)


class ChainNode:
    def __init__(self, todo: callable, next=None):
        self.todo = todo
        self.next = next

    def chain(
        self, agent, output_conversation: Conversation, act_history: cList, **kwargs
    ):
        callback = None
        if kwargs.get("callback") is not None:
            print("detected callback.")
            callback = kwargs.get("callback")
        valid_prefixes = [kwargs.get("replier_name"), kwargs.get("sender_name")]
        valid_prefixes = [x for x in valid_prefixes if x is not None]
        speak_text = remove_prefix(output_conversation.content, valid_prefixes)
        conversation_only_speak_text = Conversation(
            role=output_conversation.role,
            content=speak_text,
        )
        if self.todo is not None:
            for node_return_conversation, temp_history in self.todo(
                agent, conversation_only_speak_text, act_history, **kwargs
            ):
                if callback is not None:
                    callback(temp_history)
                yield node_return_conversation, temp_history
        if self.next is not None:
            next_conversation = (
                node_return_conversation
                if node_return_conversation is not None
                else conversation_only_speak_text
            )
            for node_return_conversation, temp_history in self.next.chain(
                agent, next_conversation, act_history, **kwargs
            ):
                if callback is not None:
                    callback(temp_history)
                yield node_return_conversation, temp_history
        if kwargs.get("logger") is not None:
            logger = kwargs.get("logger")
            changed_history = temp_history - act_history
            logger.log(changed_history)


class CommandRouter:
    def __init__(
        self,
        rules: List[Tuple[(str, Union[Self, ChainNode, callable])]],
        default: ChainNode = None,
    ):
        self.rules = rules
        self.default = default

    def chain(
        self, agent, output_conversation: Conversation, act_history: cList, **kwargs
    ):
        callback = None
        if kwargs.get("callback") is not None:
            callback = kwargs.get("callback")
        valid_prefixes = [kwargs.get("replier_name"), kwargs.get("sender_name")]
        valid_prefixes = [x for x in valid_prefixes if x is not None]

        speak_text = remove_prefix(output_conversation.content, valid_prefixes)
        conversation_only_speak_text = Conversation(
            role=output_conversation.role,
            content=speak_text,
        )
        for i, (rule_re, next) in enumerate(self.rules):
            if not isinstance(next, ChainNode) and not isinstance(next, CommandRouter):
                next = next()
            if re.match(rule_re, speak_text):
                print("router-> match: ", rule_re)
                for node_return_conversation, temp_history in next.chain(
                    agent, conversation_only_speak_text, act_history, **kwargs
                ):
                    if callback is not None:
                        callback(temp_history)
                    yield node_return_conversation, temp_history
                break
        if i == len(self.rules) - 1 and self.default is not None:
            for node_return_conversation, temp_history in self.default.chain(
                agent, conversation_only_speak_text, act_history, **kwargs
            ):
                if callback is not None:
                    callback(temp_history)
                yield node_return_conversation, temp_history
