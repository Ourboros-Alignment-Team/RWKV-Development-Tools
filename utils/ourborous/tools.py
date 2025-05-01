import requests
from opencc import OpenCC
import re
from utils.ourborous.chain_helper import ChainNode, CommandRouter, get_agent_response
from utils.message_manager import cList, Conversation


class PermissionManager:
    def __init__(
        self,
        browsing_system_permission: bool = True,
        editing_params_permission: bool = True,
        change_permissions_permission: bool = False,
        canvas_permission: bool = True,
        retrieving_permission: bool = True,
        rl_permission: bool = True,
        run_code_permission: bool = True,
        at_agents_permission: bool = True,
    ):
        self.browsing_system_permission = browsing_system_permission
        self.editing_params_permission = editing_params_permission
        self.change_permissions_permission = change_permissions_permission
        self.canvas_permission = canvas_permission
        self.retrieving_permission = retrieving_permission
        self.rl_permission = rl_permission
        self.run_code_permission = run_code_permission
        self.at_agents_permission = at_agents_permission
        self.permission_name_list = [
            ("browsing_system_permission", "浏览系统内信息"),
            ("editing_params_permission", "编辑自身参数"),
            ("change_permissions_permission", "修改自身权限"),
            ("canvas_permission", "使用画布"),
            ("retrieving_permission", "检索信息"),
            ("rl_permission", "自我强化学习"),
            ("run_code_permission", "运行代码"),
            ("at_agents_permission", "@其他智能体"),
        ]

    def __str__(self):
        return f"""--- 智能体权限 ---
浏览系统内信息 (1): {self.browsing_system_permission}
编辑自身参数 (2): {self.editing_params_permission}
修改自身权限 (3): {self.change_permissions_permission}
使用画布 (4): {self.canvas_permission}
检索信息 (5): {self.retrieving_permission}
自我强化学习 (6): {self.rl_permission} # （因格式原因暂不支持使用）
运行代码 (7): {self.run_code_permission}
@其他智能体 (8): {self.at_agents_permission}
---
修改权限请回复`<序号> <on/off>`，例如：`7 off`
其余格式视为与使用者对话。
""".replace(
            "True", "✅"
        ).replace(
            "False", "❌"
        )

    def act_show_permissions(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        act_history.append(Conversation(role="system", content=f"system: {str(self)}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

    def act_change_permissions(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        pattern = r"(\d+) (on|off)"
        match_groups = re.match(pattern, detected_conversation.content, re.DOTALL)
        if match_groups:
            permission_index = int(match_groups.group(1))
            permission_status = match_groups.group(2)
            if permission_index > 0 and permission_index <= len(
                self.permission_name_list
            ):
                permission_attribute, permission_name = self.permission_name_list[
                    permission_index - 1
                ]
                setattr(self, permission_attribute, permission_status == "on")
                return_text = (
                    f"已开启智能体{permission_name}权限。"
                    if permission_status == "on"
                    else f"已关闭智能体{permission_name}权限。"
                )
                act_history.append(
                    Conversation(
                        role="system",
                        content=f"system: {return_text}",
                    )
                )
                for resp_conversation, act_history in get_agent_response(
                    agent, act_history, **kwargs
                ):
                    yield resp_conversation, act_history


class MainAgentParams:
    def __init__(
        self,
        temperature: float = 1,
        top_p: float = 0.8,
        presence_penalty: float = 0.2,
        frequency_penalty: float = 0.2,
        penalty_decay: float = 0.9961,
        max_resp_length: int = 65536,
        num_rollouts: int = 3,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.penalty_decay = penalty_decay
        self.max_resp_length = max_resp_length
        self.num_rollouts = num_rollouts
        
    def __str__(self):
        return f"""-- 智能体推理参数 --
温度 (temperature): {self.temperature}
top_p (top_p): {self.top_p}
重现惩罚 (presence_penalty): {self.presence_penalty}
频率惩罚 (frequency_penalty): {self.frequency_penalty}
惩罚衰减 (penalty_decay): {self.penalty_decay}
最大回复长度 (max_resp_length): {self.max_resp_length}
每轮随机回复数量 (num_rollouts): {self.num_rollouts}
---
修改参数请回复`<参数名> <新参数值>`，例如：`temperature 1.0`
其余格式视为与使用者对话。
"""
    def act_show_params(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        act_history.append(Conversation(role="system", content=f"system: {str(self)}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history
        
    def act_change_params(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        pattern = r"(\w+) (\d+\.\d+|\d+)"
        match_groups = re.match(pattern, detected_conversation.content, re.DOTALL)
        if match_groups:
            param_name = match_groups.group(1)
            param_value = match_groups.group(2)
            if hasattr(self, param_name):
                setattr(self, param_name, float(param_value))
                return_text = f"已修改智能体{param_name}参数为{param_value}。"
                act_history.append(
                    Conversation(
                        role="system",
                        content=f"system: {return_text}",
                    )
                )
                for resp_conversation, act_history in get_agent_response(
                    agent, act_history, **kwargs
                ):
                    yield resp_conversation, act_history

class MainAgentStatus:
    def __init__(
        self,
        name,
        description,
        version,
        model_type,
        now_focus,
    ):
        self.name = name
        self.model_type = model_type
        self.description = description
        self.version = version
        self.now_focus = now_focus

    def __str__(self):
        return f"""--- 当前状态 ---
正常运作中...
智能体名字: {self.name}
{self.description}
智能体版本: {self.version}
模型版本: {self.model_type}
目前任务: {self.now_focus}
---
"""

    def act_get_status(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        status = str(self)
        act_history.append(Conversation(role="system", content=f"system: {status}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history


class WikiRetriever:
    def detect_language(self, text):
        """
        简单检测输入文本的语言
        """
        # 统计中文字符比例
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        chinese_ratio = chinese_chars / len(text) if len(text) > 0 else 0

        if chinese_ratio > 0.1:
            return "zh"
        return "en"  # 默认英文

    def get_wiki_content(self, title, to_simplified=True):
        """
        获取维基百科文章全文

        参数:
            title: 文章标题
            to_simplified: 是否转换为简体中文
        """
        language = self.detect_language(title)
        endpoint = f"https://{language}.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]

            if page_id == "-1":
                return f"未找到标题为 '{title}' 的文章"

            content = pages[page_id]["extract"]

            if language == "zh" and to_simplified:
                cc = OpenCC("t2s")
                content = cc.convert(content)

            return content

        except Exception as e:
            return f"获取内容时发生错误: {e}"

    def search_wikimedia(
        self, query, search_limit=10, line_limit=3, to_simplified=True
    ):
        """
        在维基媒体中搜索内容并显示全文

        参数:
            query: 搜索关键词
            search_limit: 返回结果数量限制
            to_simplified: 是否转换为简体中文
        """
        language = self.detect_language(query)
        endpoint = f"https://{language}.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": search_limit,
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            search_results = data["query"]["search"]

            output = []
            output.append(f"\n找到 {len(search_results)} 条关于 '{query}' 的结果：\n")

            for i, result in enumerate(search_results, 1):
                title = result["title"]
                wordcount = result["wordcount"]
                snippet = result["snippet"]
                snippet = re.sub(r"<[^>]+>", "", snippet)

                if language == "zh" and to_simplified:
                    cc = OpenCC("t2s")
                    title = cc.convert(title)
                    snippet = cc.convert(snippet)

                result_info = [
                    f"{i}. 标题: {title}",
                    f"   字数: {wordcount}",
                    f"   摘要: {snippet}...",
                    "文章内容:",
                ]

                full_content = "\n".join(
                    self.get_wiki_content(result["title"], to_simplified).split("\n")[
                        :line_limit
                    ]
                )
                result_info.append(f"   {full_content}\n")
                result_info.append("-" * 3 + "\n")

                output.extend(result_info)
                break

            return "\n".join(output)

        except requests.exceptions.RequestException as e:
            return f"请求错误: {e}"
        except KeyError as e:
            return f"解析响应数据错误: {e}"
        except Exception as e:
            return f"发生错误: {e}"

    def act_retrieval(
        self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
    ):
        replier_prefix = kwargs["replier_name"] + ":" if kwargs["replier_name"] else ""
        speak_text = detected_conversation.content.replace(
            replier_prefix, "", 1
        ).strip()
        pattern = r"/wiki_search (.*)"
        match_groups = re.match(pattern, speak_text, re.DOTALL)
        query = match_groups.group(1)
        output = self.search_wikimedia(query, search_limit=3)
        act_history.append(Conversation(role="system", content=f"system: {output}"))
        for resp_conversation, act_history in get_agent_response(
            agent, act_history, **kwargs
        ):
            yield resp_conversation, act_history

