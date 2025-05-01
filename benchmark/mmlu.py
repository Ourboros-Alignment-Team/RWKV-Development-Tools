import requests
import json
from utils.message_manager import cList, Conversation
import os
import csv
import gc,torch
from config import global_config

class MMLUBenchmark:
    def __init__(self, agent, mmlu_folder, bmk_type="mmlu",constraint_output=True):
        assert bmk_type in ["mmlu","ceval","cmmlu"]
        self.mmlu_folder = mmlu_folder
        self.agent = agent
        self.type = bmk_type
        self.constraint_output=constraint_output
        
    def run_benchmark(
        self,
        temp=1,
        top_p=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        decay_penalty=0.9961,
        resp_start_with=None
    ):
        files = os.listdir(self.mmlu_folder)
        total = len(files)
        n_questions = 0
        correct_questions = 0
        for i, file in enumerate(files, 0):
            if file.lower().endswith(".csv"):
                name=os.path.basename(file).split("_test.csv")[0]
                full_dir = os.path.join(self.mmlu_folder, file)
                with open(full_dir, "r", encoding="utf-8") as f:
                    csv_reader = csv.reader(f)
                    answer=None
                    for j,row in enumerate(csv_reader):
                        if self.type in ["ceval", "cmmlu"]:
                            if j==0:
                                continue
                        if self.type in ["mmlu"]:
                            question, ch_a, ch_b, ch_c, ch_d, answer = row
                        if self.type in ["cmmlu"]:
                            q_id,question, ch_a, ch_b, ch_c, ch_d, answer = row
                        elif self.type in ["ceval"]:
                            q_id,question, ch_a, ch_b, ch_c, ch_d = row
                        prompt = f"question: {question}\nA.{ch_a}\nB.{ch_b}\nC.{ch_c}\nD.{ch_d}"
                        conversations = cList([Conversation("conversation", prompt)])
                        resps = ""
                        for resp in self.send_message(
                            conversations,
                            temp,
                            top_p,
                            presence_penalty,
                            frequency_penalty,
                            decay_penalty,
                            resp_start_with,
                            " <A/B/C/D>" if self.constraint_output else None
                        ):
                            resps += resp
                        first_letter = None

                        for i, char in enumerate(resps):
                            if (
                                char in ["A", "B", "C", "D"]
                                and (i == 0 or not resps[i - 1].isalpha())
                                and (i == len(resps) - 1 or not resps[i + 1].isalpha())
                            ):
                                first_letter = char
                                break
                        is_correct = first_letter == answer if (answer is not None and first_letter is not None) else False
                        n_questions += 1
                        correct_questions += int(is_correct)

                        now_score = correct_questions / n_questions
                        if self.type in ["mmlu", "cmmlu"]:
                            yield conversations(), f"answer: {resps}", answer, first_letter, is_correct, now_score, i / total
                        elif self.type in ["ceval"]:
                            yield conversations(), f"answer: {resps}", answer, first_letter, q_id,name, i / total
                gc.collect()
                torch.cuda.empty_cache()
                        
    def send_message(
        self,
        conversations,
        temp=1,
        top_p=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        decay_penalty=0.9961,
        resp_start_with_str=None,
        format_constrain_str=None
    ):
        resp_start_with_str=resp_start_with_str if resp_start_with_str else f"{self.agent.replier}:"
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
            "format_constrain_str":format_constrain_str
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
