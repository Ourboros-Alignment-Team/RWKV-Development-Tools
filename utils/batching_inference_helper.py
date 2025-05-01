from config import global_config

infer_config = global_config.infer_service_config

from RWKV.functions import sample_logits, ppl, batch_block_infer
import torch
from utils.collections import parse_format_constrain_str
import threading
from collections import OrderedDict
from typing import List, Tuple
import traceback
import time

class CollateInferenceTokens:
    def __init__(self):
        self.token_state_dict = OrderedDict()

    def update(self, idx, token: int, state):
        self.token_state_dict[idx] = (token, state)

    @property
    def batch(self):
        idx_list = list(self.token_state_dict.keys())
        token_state_list = list(self.token_state_dict.values())  # [(token, states)]
        states = token_state_list[0][1]
        tokens = [[token_state_list[0][0]]]
        for token, state in token_state_list[1:]:
            # 需要处理state为None的情况
            states = states + state
            tokens.append([token])
        return idx_list, tokens, states


class ConstraintGenerateBlockList:
    def __init__(self, task, constraint_str):
        if constraint_str is None:
            self.constraint_blocks = [
                ConstraintGenerateBlock(
                    "generate", max_tokens=task.max_tokens, token_stop=task.token_stop
                )
            ]
        else:
            try:
                str_blocks = parse_format_constrain_str(constraint_str)
                self.constraint_blocks = []
                for block in str_blocks:
                    if block["type"] == "str":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock("str", text=block["text"])
                        )
                    elif block["type"] == "select":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock(
                                "select", selections=block["selections"]
                            )
                        )
                    elif block["type"] == "generate":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock(
                                "generate",
                                max_tokens=block["n_max"],
                                token_stop=block["stop"],
                            )
                        )
            except Exception as e:
                traceback.print_exc()
                print(f"任务解析约束字符串失败，使用默认生成方式。错误信息：{e}")
                self.constraint_blocks = [
                    ConstraintGenerateBlock(
                        "generate",
                        max_tokens=task.max_tokens,
                        token_stop=task.token_stop,
                    )
                ]
        self.now_block = self.constraint_blocks[0]

    def next(self, last_tokens):
        # allow_tokens 为None表示不限制
        over = False
        block_over = self.now_block.next(last_tokens)
        if block_over:
            idx = self.constraint_blocks.index(self.now_block)
            if idx + 1 < len(self.constraint_blocks):
                self.now_block = self.constraint_blocks[idx + 1]
            else:
                over = True
        return over


class ConstraintGenerateBlock:
    def __init__(self, rule, **kwargs):
        self.allow_tokens = None
        self.direct_infer_tokens = None
        self.rule = rule
        if rule == "generate":
            self.max_tokens = kwargs["max_tokens"]
            self.token_stop = kwargs["token_stop"]
            self.n_now = 0
        elif rule == "select":
            self.selections = kwargs["selections"]
            self.selections = [
                global_config.tokenizer_eval.encode(x)
                for x in self.selections
                if x.strip()
            ]
            self.allow_tokens = [s[0] for s in self.selections]
        elif rule == "str":
            self.text = kwargs["text"]
            self.text_tokens = global_config.tokenizer_eval.encode(self.text)
            self.direct_infer_tokens = self.text_tokens

    def next(self, last_tokens):
        over = False
        if self.rule == "generate":
            self.n_now += len(last_tokens)
            if self.n_now >= self.max_tokens or last_tokens[-1] in self.token_stop:
                over = True
            self.allow_tokens = None
            self.direct_infer_tokens = None
            return over
        elif self.rule == "select":
            self.selections = [
                s[1:]
                for s in self.selections
                if s and last_tokens[-1] == s[0] and s[1:]
            ]
            self.allow_tokens = [s[0] for s in self.selections]
            self.direct_infer_tokens = None
            return len(self.selections) == 0
        elif self.rule == "str":
            self.allow_tokens = None
            self.direct_infer_tokens = self.text_tokens
            return True


class BatchingTask:
    def __init__(
        self,
        state,
        begin_with_tokens,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
        penalty_decay,
        constraint_str,
        token_stop,
        token_stop_supp,
        max_tokens,
        token_ban=[],
        occurence={},
        callback: callable = None,
    ):
        self.begin_with_tokens = begin_with_tokens
        self.temp = temp
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.penalty_decay = penalty_decay
        self.constraint_str = constraint_str
        self.token_stop = token_stop
        self.token_stop_supp = token_stop_supp
        self.max_tokens = max_tokens
        self.token_ban = token_ban
        self.callback = callback

        self.state = state
        self.in_ppls = []
        self.out_ppls = []
        self.resp_tokens = []
        self.iter_tokens = []
        self.last_out = None
        self.over = False

        self.rule_blocks = self.init_rule_blocks()
        self.occurence = occurence

    def reach_max_tokens(self):
        return len(self.resp_tokens) >= self.max_tokens

    def init_rule_blocks(self):
        return ConstraintGenerateBlockList(self, self.constraint_str)


class BatchingInferenceHelper:
    def __init__(
        self,
        rwkv,
        max_bsz: int = 5,
    ):
        self.model = rwkv
        self.batch_tasks = []
        self.wait_for_inference_tasks = []  # state_id, tokens_list
        self.max_bsz = max_bsz
        self.tokenizer = global_config.tokenizer_eval
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def init_task(self, task: BatchingTask):
        tokens, states = task.begin_with_tokens, task.state
        out, new_states = batch_block_infer(self.model, [tokens], states)
        task.in_ppls = ppl(tokens[1:], out[0, :-1, :])
        task.state = new_states
        task.last_out = out
        return task

    def _loop(self):
        while True:
            time.sleep(0.001)
            if len(self.batch_tasks) < self.max_bsz and self.wait_for_inference_tasks:
                new_task = self.wait_for_inference_tasks.pop(0)
                self.batch_tasks.append(self.init_task(new_task))

            self.step()

    @torch.no_grad()
    def step(self):
        if self.batch_tasks:
            self.batch_tasks: List[BatchingTask]
            batch_tokens = CollateInferenceTokens()
            # 加载初始token与单步sample
            for i, task in enumerate(self.batch_tasks):
                task: BatchingTask
                if task.rule_blocks.now_block.direct_infer_tokens:
                    direct_infer_tokens = task.rule_blocks.now_block.direct_infer_tokens
                    task.resp_tokens += direct_infer_tokens
                    task.iter_tokens = direct_infer_tokens
                    out, new_states = batch_block_infer(
                        self.model, [direct_infer_tokens], task.state
                    )
                    task.out_ppls += ppl(direct_infer_tokens[1:], out[0, :-1, :])
                    task.state = new_states
                    task.last_out = out
                    thread = threading.Thread(
                        target=task.callback,
                        args=(
                            task.iter_tokens,
                            task.state,
                            task.in_ppls,
                            task.out_ppls,
                            task.over,
                        ),
                        daemon=True,
                    )
                    thread.start()
                else:
                    token, token_ppl = self.sample_logits(
                        task,
                    )  # 要不要换成block infer
                    task.resp_tokens.append(token)
                    task.out_ppls += token_ppl
                    task.iter_tokens = [token]
                    batch_tokens.update(i, token, task.state)

                task.over = task.rule_blocks.next(task.iter_tokens)

            if batch_tokens.token_state_dict:
                idx_list, tokens, states = batch_tokens.batch
                batch_out, new_states = batch_block_infer(
                    self.model,
                    tokens_batches=tokens,
                    state=states,
                )
                new_states_list = new_states.unbind()
                out_list: List[torch.Tensor] = torch.unbind(batch_out, dim=0)
                out_list = [x.unsqueeze(0) for x in out_list]

                for idx, out, state in zip(idx_list, out_list, new_states_list):
                    task: BatchingTask = self.batch_tasks[idx]
                    task.last_out = out
                    task.state = state
                    if task.over:
                        if task.token_stop_supp:
                            _, task.state = batch_block_infer(
                                self.model, [task.token_stop_supp], task.state
                            )
                        thread = threading.Thread(
                            target=task.callback,
                            args=(
                                [],
                                task.state,
                                task.in_ppls,
                                task.out_ppls,
                                task.over,
                            ),
                            daemon=True,
                        )
                        thread.start()
                    else:
                        thread = threading.Thread(
                            target=task.callback,
                            args=(
                                task.iter_tokens,
                                task.state,
                                task.in_ppls,
                                task.out_ppls,
                                task.over,
                            ),
                            daemon=True,
                        )
                        thread.start()
            for task in self.batch_tasks:
                if task.over:
                    self.batch_tasks.remove(task)

    def sample_logits(
        self,
        task: BatchingTask,
    ):
        out = task.last_out
        occurrence = task.occurence
        for n in occurrence:
            out[0, -1, n] -= (
                task.presence_penalty + occurrence[n] * task.frequency_penalty
            )
        if task.rule_blocks.now_block.allow_tokens:
            allow_tokens = task.rule_blocks.now_block.allow_tokens
            mask = torch.full_like(out[0, -1, :], -1e38, device=out.device)
            mask[allow_tokens] = out[0, -1, allow_tokens]
            out[0, -1, :] = mask
        for ban_token in task.token_ban:
            out[0, -1, ban_token] = torch.full_like(
                out[0, -1, ban_token], -1e38, device=out.device
            )
        token = sample_logits(
            out[0, -1, :],
            temperature=task.temp,
            top_p=task.top_p,
        )
        token_ppl = ppl([token], out[0, -1:, :])
        for xxx in occurrence:
            occurrence[xxx] *= task.penalty_decay
        if 49 <= token <= 58:  # numbers
            pass
        elif token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        task.occurence = occurrence
        return token, token_ppl


class BatchingInferenceManager:
    def __init__(self, rwkv, max_bsz: int = 5):
        self.helper = BatchingInferenceHelper(rwkv, max_bsz)
        self.lock = threading.Lock()
        self.result = None
        self.over = False

    def run_task(
        self,
        begin_with_tokens: List[int],
        state,
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        penalty_decay: float,
        constraint_str: str,
        token_stop: List[int],
        token_stop_supp: List[int],
        max_tokens: int,
        token_ban: List[int],
        occurence: dict,
    ):
        event = threading.Event()

        def inside_callback(resp_tokens, state, in_ppls, out_ppls, over):
            with self.lock:
                self.result = resp_tokens, state, in_ppls, out_ppls
                self.over = over
            event.set()

        task = BatchingTask(
            state=state,
            begin_with_tokens=begin_with_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            constraint_str=constraint_str,
            token_stop=token_stop,
            token_stop_supp=token_stop_supp,
            max_tokens=max_tokens,
            token_ban=token_ban,
            occurence=occurence,
            callback=inside_callback,
        )
        self.helper.wait_for_inference_tasks.append(task)
        while True:
            event.wait()
            with self.lock:
                resp_tokens, state, in_ppls, out_ppls = self.result
                yield resp_tokens, state, in_ppls, out_ppls, self.over
                if self.over:
                    event.clear()
                    break
            event.clear()
