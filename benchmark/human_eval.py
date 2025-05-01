from .human_eval_bmk.human_eval.data import stream_jsonl
from .human_eval_bmk.human_eval.evaluate_functional_correctness import (
    entry_point as eval_correctness,
)
from utils.message_manager import cList, Conversation
import requests
import json
import re
from config import global_config

class HumanEvalBenchmark:
    def __init__(self, agent, human_eval_dir, constraint_output=True):
        self.human_eval_dir = human_eval_dir
        self.agent = agent
        self.constraint_output = constraint_output

    def run_benchmark(
        self,
        temp=1,
        top_p=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        decay_penalty=0.9961,
        prompt_start_with=None,
        resp_start_with=None,
    ):
        for bmk_task_dict in stream_jsonl(self.human_eval_dir):
            task_id, q = bmk_task_dict["task_id"], bmk_task_dict["prompt"]
            prompt_start_with = (
                prompt_start_with if prompt_start_with else f"{self.agent.sender}: "
            )
            resp_start_with_str = (
                resp_start_with if resp_start_with else f"{self.agent.replier}:"
            )
            conversations = cList(
                [Conversation("conversation", f"{prompt_start_with}{q}")]
            )
            resps = ""
            for resp in self.send_message(
                conversations,
                temp,
                top_p,
                presence_penalty,
                frequency_penalty,
                decay_penalty,
                resp_start_with_str,
                " ```python$1000->65535->11$" if self.constraint_output else None,
            ):
                resps += resp

            yield conversations(), f"{resp_start_with_str}{resps}", task_id

    def extract_code(self, resp_str):
        code_block_pattern = r"```(?:python|[\w]+)?\s*([\s\S]+?)```"

        # Search for code block in the response string
        match = re.search(code_block_pattern, resp_str)

        if match:
            # Extract and return the code inside the matched block
            return match.group(
                1
            ).strip()  # Strip unnecessary whitespace from the code block
        else:
            # If no code block is found, return an empty string
            return ""

    def evaulate_acc_from_log_file(self, log_fp):
        results = eval_correctness(log_fp, problem_file=self.human_eval_dir)
        return results

    def send_message(
        self,
        conversations,
        temp=1,
        top_p=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        decay_penalty=0.9961,
        resp_start_with_str=None,
        format_constrain_str=None,
    ):
        package = {
            "conversations": conversations.to_dict_list(),
            "resp_start_with_role": "response",
            "resp_start_with_str": resp_start_with_str,
            "stop_with_tokens": global_config.role["response"]["postfix"][:1],
            "stop_supp_tokens": [],
            "temp": temp,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "decay_penalty": decay_penalty,
            "use_now_state_idx": None,
            "save_to_now_state_idx": None,
            "max_resp_len": 500,
            "format_constrain_str": format_constrain_str,
        }
        with requests.post(
            self.agent.infer_server + "/chat_task", json=package, stream=True
        ) as response:
            if response.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"HTTP request failed with status code {response.status_code}"
                )

            else:
                for chunk in response.iter_lines():

                    result = json.loads(chunk)
                    res_text = result["next"]
                    yield res_text
