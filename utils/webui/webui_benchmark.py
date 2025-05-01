import requests
import json

# from utils.qa_benchmark import prompts, p_bot_name, p_usr_name
from utils.message_manager import cList, Conversation
from benchmark.mmlu import MMLUBenchmark
from benchmark.human_eval import HumanEvalBenchmark
import os
from config import global_config
import concurrent.futures
import time

# 全局变量用于控制基准测试状态
is_stopped = False


def run_benchmark(
    agent,
    question_list,
    temp=1,
    top_p=0.7,
    presence_penalty=0.2,
    frequency_penalty=0.2,
    penalty_decay=0.9961,
    using_init_state=False,
    init_state_dir=None,
):
    # TODO: 改成并行
    global is_stopped
    results = []

    for question_dict in question_list:
        q_role, question_text, resp_role, resp_prefix = (
            question_dict["quesetion_role"],
            question_dict["quesetion_text"],
            question_dict["response_role"],
            question_dict["response_text_prefix"],
        )
        if is_stopped:
            is_stopped = False
            break

        state_id = None
        if init_state_dir is not None and using_init_state:
            package = {
                "load_dir": init_state_dir,
            }
            resp = requests.post(
                agent.infer_server + "/regist_state_id", json=package
            ).json()
            state_id = resp["access_token"]

        results.append(
            f"\n==========================================\n{question_text}\n"
        )

        q = cList([Conversation(q_role, question_text)])

        package = {
            "conversations": q.to_dict_list(),
            "resp_start_with_role": resp_role,
            "resp_start_with_str": resp_prefix,
            "stop_with_tokens": global_config.role[resp_role]["postfix"][:1],
            "stop_supp_tokens": [],
            "temp": temp,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "decay_penalty": penalty_decay,
            "use_now_state_idx": state_id,
            "save_to_now_state_idx": None,
            "max_resp_len": 500,
        }
        results.append(resp_prefix)
        with requests.post(
            agent.infer_server + "/chat_task", json=package, stream=True
        ) as response:
            if response.status_code != 200:
                results.append(
                    f"系统: Error: Received status code {response.status_code}"
                )
            else:
                for chunk in response.iter_lines():
                    if is_stopped:
                        is_stopped = False
                        break

                    result = json.loads(chunk)
                    res_text = result["next"]
                    results.append(f"{res_text}")
                    yield "".join(results)


def stop_benchmark():
    global is_stopped
    is_stopped = True


def run_mmlu(
    agent,
    save_output_fp,
    mmlu_dir,
    temp,
    top_p,
    presence_penalty,
    frequency_penalty,
    penalty_decay,
    resp_start_with,
    bmk_type,
    constraint_output,
):
    B = MMLUBenchmark(agent, mmlu_dir, bmk_type, constraint_output)
    total_str = ""
    if bmk_type in ["ceval"]:
        save_json = {}

        for i, (
            req_conversations,
            resp_str,
            ground_truth,
            answer,
            q_id,
            name,
            now_progress,
        ) in enumerate(
            B.run_benchmark(
                temp,
                top_p,
                presence_penalty,
                frequency_penalty,
                penalty_decay,
                resp_start_with,
            )
        ):
            total_str
            if name not in save_json:
                save_json[name] = {}
                save_json[name][str(q_id)] = answer
            else:
                save_json[name][str(q_id)] = answer
            if i % 20 == 0:
                yield total_str, "不支持", now_progress
        yield total_str, "不支持", now_progress

        with open(save_output_fp, "w", encoding="utf-8") as f:
            f.write(json.dumps(save_json, ensure_ascii=False))

    else:
        for i, (
            req_conversations,
            resp_str,
            ground_truth,
            answer,
            is_correct,
            now_score,
            now_progress,
        ) in enumerate(
            B.run_benchmark(
                temp,
                top_p,
                presence_penalty,
                frequency_penalty,
                penalty_decay,
                resp_start_with,
            )
        ):
            total_str += f"==============================================\n{req_conversations}\n{resp_str}\n正确答案：{ground_truth}\n回复：{answer}\n{is_correct}\n"
            if i % 20 == 0:
                yield total_str, now_score, now_progress
        yield total_str, now_score, now_progress
        with open(save_output_fp, "w", encoding="utf-8") as f:
            f.write(total_str)


def run_humaneval(
    agent,
    save_output_dir,
    humaneval_dir,
    temp,
    top_p,
    presence_penalty,
    frequency_penalty,
    penalty_decay,
    req_start_with,
    resp_start_with,
    constraint_output,
):
    B = HumanEvalBenchmark(agent, humaneval_dir, constraint_output)
    total_str = ""
    save_dicts = []
    for i, (req_conversations, resp_str, q_id) in enumerate(
        B.run_benchmark(
            temp,
            top_p,
            presence_penalty,
            frequency_penalty,
            penalty_decay,
            req_start_with,
            resp_start_with,
        )
    ):
        save_dict = {"task_id": q_id, "completion": B.extract_code(resp_str)}
        save_dicts.append(save_dict)
        total_str += f"==============================================\n{req_conversations}\n{resp_str}\n"
        yield total_str
    log_fp = os.path.join(save_output_dir, "humaneval_log.jsonl")
    with open(log_fp, "w", encoding="utf-8") as f:
        for d in save_dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    results = B.evaulate_acc_from_log_file(log_fp)
    res_fp = os.path.join(save_output_dir, "humaneval_result.jsonl")
    with open(res_fp, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False))
    total_str += f"==============================================\n结果已保存\n"


def save_question_list(question_list, save_dir):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_fp = os.path.join(save_dir, f"question_list_{time_str}.json")
    with open(save_fp, "w", encoding="utf-8") as f:
        f.write(json.dumps(question_list, ensure_ascii=False))
