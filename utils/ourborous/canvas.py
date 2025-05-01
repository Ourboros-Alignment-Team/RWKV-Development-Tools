from utils.ourborous.chain_helper import ChainNode, CommandRouter, get_agent_response
from utils.message_manager import cList, Conversation
import re


class OurborousCanvas:
    def __init__(self):
        self.txt = ""
        self.help_text = """指令一览：
* 查看画布内容 /show  * 结束画布交互: /end  * 清空画布: /clear  * 在末尾增加内容 /add <内容> * 清空并写入内容: /write <内容>
* 在指定行插入内容: /insert <行号> <内容>  * 删除指定行内容: /delete <行号>  * 修改指定行内容: /edit <行号> <内容>"""

        self.router = CommandRouter(
            rules=[],
            default=ChainNode(todo=self.act_failed, next=None),
        )
        self.router.rules = [
            (r"/show", ChainNode(todo=self.act_serialize, next=self.router)),
            (r"/end", ChainNode(todo=self.act_end, next=None)),
            (r"/clear", ChainNode(todo=self.act_cls, next=self.router)),
            (r"/write (.*)", ChainNode(todo=self.act_write, next=self.router)),
            (r"/add (.*)", ChainNode(todo=self.act_add, next=self.router)),
            (
                r"/insert (\d+) (.*)",
                ChainNode(todo=self.act_insert, next=self.router),
            ),
            (r"/delete (\d+)", ChainNode(todo=self.act_delete, next=self.router)),
            (r"/edit (\d+) (.*)", ChainNode(todo=self.act_edit, next=self.router)),
        ]
        self.entry = ChainNode(todo=self.act_serialize, next=self.router)

    @property
    def lines(self):
        return self.txt.split("\n")

    def act_failed(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        act_history.append(
            Conversation(
                role="system",
                content=f"system: 未匹配到指令，已返回对话。",
            )
        )
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def serialize(self):
        lines = self.lines
        output = ""
        for i, line in enumerate(lines, 1):
            output += f"行{i}:{line}\n"
        return output

    def act_serialize(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        output = self.serialize()

        act_history.append(
            Conversation(
                role="system",
                content=f"""画布内容:
{output}
{self.help_text}""",
            )
        )
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_end(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        act_history.append(
            Conversation(
                role="system", content=f"system: 已结束与画布的交互，返回对话。"
            )
        )
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_edit(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):

        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/edit (\d+) (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        line_index = int(match_groups.group(1))
        new_line = match_groups.group(2)
        output = self.edit(line_index, new_line)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_delete(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/delete (\d+)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        line_index = int(match_groups.group(1))
        output = self.delete(line_index)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_insert(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/insert (\d+) (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        line_index = int(match_groups.group(1))
        new_line = match_groups.group(2)
        output = self.insert(line_index, new_line)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_add(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/add (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        text = match_groups.group(1)
        output = self.add(text)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_write(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/write (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        text = match_groups.group(1)
        output = self.write(text)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history
        

    def act_cls(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        output = self.clear()
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def __len__(self):
        return len(self.lines)

    def edit(self, line_index, new_line):
        if line_index < 1 or line_index > len(self):
            return f"错误：行号超出范围，请输入{1}~{len(self)}之间的行号。"
        lines = self.lines
        lines[line_index - 1] = new_line
        self.txt = "\n".join(lines)
        return f"行{line_index}已修改为：{new_line}\n{self.help_text}"

    def clear(self):
        self.txt = ""
        return f"画布已清空。\n{self.help_text}"

    def add(self, text):
        self.txt += text
        return f"已在画布末尾写入内容，目前:\n{self.serialize()}"

    def write(self, text):
        self.txt = text
        return f"已清空画布并写入内容，目前:\n{self.serialize()}"

    def insert(self, line_index, text):
        if line_index < 1 or line_index > len(self) + 1:
            return f"错误：行号超出范围，请输入{1}~{len(self)+1}之间的行号。"
        lines = self.lines
        lines.insert(line_index - 1, text)
        self.txt = "\n".join(lines)
        return f"已在第{line_index}行插入内容{text}。\n{self.help_text}"

    def delete(self, line_index):
        if line_index < 1 or line_index > len(self):
            return f"错误：行号超出范围，请输入{1}~{len(self)}之间的行号。"
        lines = self.lines
        del lines[line_index - 1]
        self.txt = "\n".join(lines)
        return f"已删除第{line_index}行内容。\n{self.help_text}"
