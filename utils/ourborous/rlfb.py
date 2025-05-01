# from utils.ourborous.chain_helper import ChainNode, CommandRouter, get_agent_response
# from utils.message_manager import cList, Conversation
# from utils.ourborous.history import OurborousAgentHistoryContainer
# from config import global_config
# import re


# class ReinforceLearningFeedback:
#     def __init__(self):
#         self.now_index = 0

#     def act_show_turn(
#         self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
#     ):
#         if len(agent.history_container.untrained_histories) == 0:
#             act_history.append(
#                 Conversation(
#                     role="system",
#                     content=f"system: 没有可训练的历史记录。",
#                 )
#             )
#             for resp_conversation, act_history in get_agent_response(
#                 agent, act_history, **kwargs
#             ):
#                 yield resp_conversation, act_history
#         else:
#             choises = agent.history_container.untrained_histories[self.now_index]
#             choises_str = f"-- 第{self.now_index+1}轮/共{len(agent.history_container.untrained_histories)}轮对话 --\n"
#             for i, choise_conversations in enumerate(choises):
#                 choise_clist = cList()
#                 for conversation in choise_conversations:
#                     if conversation.role in global_config.ego_types:
#                         conversation_str = f"{conversation.content}"
#                         choise_clist.append(
#                             Conversation(role=conversation.role, content=conversation_str)
#                         )
#                     else:
#                         conversation_str = f"{conversation.role}: {conversation.content}"
#                 # choises_str += f"第{i+1}个回复: \n{choise_clist()}\n"

#             choises_str += (
#                 "--- 可选操作 ---\n"
#                 + ("回复 `←` 切换到上一轮对话\n" if self.now_index > 0 else "")
#                 + (
#                     "回复 `→` 切换到下一轮对话\n"
#                     if self.now_index < len(agent.history_container.untrained_histories) - 1
#                     else ""
#                 )
#                 + f"""回复 `<序号>: <分数>; <序号>: <分数>...` 可以标记每个回复的分数（最大值为1，最小值为-1）。
#     例如：
#     `1: 0.5; 2: 0.7; 3: -0.3`
#     表示将第1个回复的分数标记为0.5，第2个回复的分数标记为0.7，第3个回复的分数标记为-0.3。
#     其余回复视为和使用者对话。
#     """
#             )
#             act_history.append(
#                 Conversation(
#                     role="system",
#                     content=choises_str,
#                 )
#             )
#             for resp_conversation, act_history in get_agent_response(
#                 agent, act_history, **kwargs
#             ):
#                 yield resp_conversation, act_history

#     def act_opt_turn(
#         self, agent, detected_conversation: Conversation, act_history: cList, **kwargs
#     ):
#         pattern_left = r"←"
#         pattern_right = r"→"
#         pattern_score = r"(\d+):\s*(-?\d+(?:\.\d+)?)"
#         match_left = re.match(pattern_left, detected_conversation.content, re.DOTALL)
#         match_right = re.match(pattern_right, detected_conversation.content, re.DOTALL)
#         match_scores = re.findall(
#             pattern_score, detected_conversation.content, re.DOTALL
#         )
#         if match_left:
#             self.now_index -= 1
#             if self.now_index < 0:
#                 self.now_index = 0
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 已经是第一轮对话，无法继续切换，已返回和使用者的对话。",
#                     )
#                 )
#                 for resp_conversation, act_history in get_agent_response(
#                     agent, act_history, **kwargs
#                 ):
#                     yield resp_conversation, act_history
#             else:
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 切换到上一轮。",
#                     )
#                 )
#                 for resp_conversation, act_history in ChainNode(
#                     todo=self.act_show_turn, next=None
#                 ).chain(
#                     agent, detected_conversation, act_history, **kwargs
#                 ):
#                     yield resp_conversation, act_history
#         elif match_right:
#             self.now_index += 1
#             if self.now_index >= len(agent.history_container.untrained_histories):
#                 self.now_index = len(agent.history_container.untrained_histories) - 1
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 已经是最后一轮对话，无法继续切换，已返回和使用者的对话。",
#                     )
#                 )
#                 for resp_conversation, act_history in get_agent_response(
#                     agent, act_history, **kwargs
#                 ):
#                     yield resp_conversation, act_history
#             else:
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 切换到下一轮。",
#                     )
#                 )
#                 for resp_conversation, act_history in ChainNode(
#                     todo=self.act_show_turn, next=None
#                 ):
#                     yield resp_conversation, act_history
#         elif match_scores:
#             return_text="system:"
#             turn_choises = agent.history_container.untrained_histories[self.now_index]
#             for index, score in match_scores: 
#                 if int(index) > len(turn_choises):
#                     return_text += f"第{index}个回复不存在，无法调整为{score}。\n"
#                     continue
#                 else:
#                     turn_choises[int(index) - 1][1] = float(score)
#                     return_text += f"第{index}个回复的分数已标记为{score}。\n"
#             act_history.append(
#                 Conversation(
#                     role="system",
#                     content=return_text,
#                 )
#             )
#             if self.now_index < len(agent.history_container.untrained_histories) - 1:
#                 self.now_index += 1
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 自动切换到下一轮，可以选择继续标记或和使用者对话。",
#                     )
#                 )
#                 for resp_conversation, act_history in ChainNode(
#                     todo=self.act_show_turn, next=None
#                 ):
#                     yield resp_conversation, act_history
#             else:
#                 act_history.append(
#                     Conversation(
#                         role="system",
#                         content="system: 已经完成最后一轮对话的标记，返回和使用者的对话。",
#                     )
#                 )
#                 for resp_conversation, act_history in get_agent_response(
#                     agent, act_history, **kwargs
#                 ):
#                     yield resp_conversation, act_history
                
                
