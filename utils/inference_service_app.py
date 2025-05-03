from config import global_config
import os

infer_config = global_config.infer_service_config
from utils.message_manager import Conversation, cList
from utils.collections import parse_format_constrain_str
from config import RWKVInfer as RWKV
from typing import List
from RWKV.functions import sample_logits, ppl, batch_block_infer, batch_chat
from config import RWKVStates
import torch
import sys, gc
import uuid
import json
import re
from utils import batching_inference_helper
from utils.containers import LocalMemoryStatePool

from utils.batching_inference_helper import BatchingInferenceManager
import psutil
import os


class InferenceAPP:
    def __init__(self):
        gc.collect()
        self.tokenizer = global_config.tokenizer_eval
        self.model = RWKV(infer_config)

        self.states_pool = LocalMemoryStatePool(cache_dir=global_config.cache_dir)
        self.res_buffer = {}
        self.broadcast_host = global_config.server_config.infer.batching_broadcast_host
        self.broadcast_port = global_config.server_config.infer.batching_broadcast_port

        self.batching_manager = BatchingInferenceManager(
            self.model,
            infer_config.batching_batch_size,
        )

    def batch_block_infer(
        self, tokens_batches: list, state: RWKVStates, chunk_len: int = 512
    ):
        return batch_block_infer(self.model, tokens_batches, state, chunk_len)

    def regist_state_id(self, load_dir: str = None):
        state_id = "sk-" + str(uuid.uuid4())
        self.states_pool[state_id] = (
            torch.load(load_dir, map_location=next(self.model.parameters()).device)
            if load_dir
            else None
        )
        return state_id

    def save_state(self, state_id: str, to_dir: str):
        if state_id in self.states_pool:
            self.states_pool.save(state_id, to_dir)

    def load_state(self, state_id: str, load_dir: str = None):
        try:
            self.states_pool[state_id] = torch.load(load_dir) if load_dir else None
        except:
            print(f"load state: state not found")
            self.states_pool[state_id] = None
        print(f"load state id={state_id} from {load_dir}.")

    def remove_state_id(self, state_id: str):
        if state_id in self.states_pool.keys():
            del self.states_pool[state_id]

    def copy_state(self, state_id: str, target_state_id: str):
        if state_id in self.states_pool.keys():
            self.states_pool[target_state_id] = self.states_pool[state_id]

    def infer_batch(
        self,
        tokens_batches: list,
        state: RWKVStates = None,
        latent_output=False,
        save_cache_dir=None,
    ):
        if latent_output:
            out, state, latent_out = self.model(tokens_batches, state, latent_output)
            if save_cache_dir:
                cache_dict = {
                    "logits": out.detach().cpu(),
                    "latent_out": latent_out.detach().cpu(),
                }
                torch.save(cache_dict, save_cache_dir)
            return out, state, latent_out
        out, state = self.model(tokens_batches, state)
        if save_cache_dir:
            cache_dict = {
                "logits": out.detach().cpu(),
            }
            torch.save(cache_dict, save_cache_dir)
        return out, state

    def block_infer(self, tokens, state, chunk_len=512):
        out = None
        tokens = [int(x) for x in tokens]
        while len(tokens) > 0:
            out, state = self.model.infer(tokens[:chunk_len], state)
            tokens = tokens[chunk_len:]
        return out, state

    def chat(
        self,
        conversations: cList,
        resp_start_with_tokens: List[int],
        stop_with_tokens: List[int],
        stop_supp_tokens: List[int],
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        decay_penalty: float,
        use_now_state_idx=None,
        save_to_now_state_idx=None,
        chunk_len: int = 512,
        max_resp_len: int = 512,
        stream=False,
        stream_chunk=9,
        format_constrain_str=None,
        token_out=False,
        token_ban=[],
        need_ppl=False,
    ):
        in_ppls = []
        state = self.states_pool[use_now_state_idx] if use_now_state_idx else None
        upload_tokens = (
            conversations.to_tokens(self.tokenizer.encode)[0] + resp_start_with_tokens
            if conversations
            else resp_start_with_tokens if resp_start_with_tokens else [0]
        )
        out, new_state = self.block_infer(
            upload_tokens,
            state,
            chunk_len,
        )
        if len(upload_tokens) > 1:
            in_ppls += ppl(upload_tokens[1:], out[0, :-1, :])
        occurrence = {}
        tokens = []
        start_index = 0
        count = 0
        out_ppls = []
        if format_constrain_str is not None:
            blocks = parse_format_constrain_str(format_constrain_str)
            for block in blocks:
                if block["type"] == "str":
                    cl = cList()
                    txt = block["text"]

                    c = Conversation(role="text", content=txt)
                    cl.append(c)
                    out, new_state = self.block_infer(
                        cl.to_tokens(self.tokenizer.encode)[0],
                        new_state,
                        chunk_len,
                    )
                    if need_ppl:
                        out_ppls += ppl(
                            cl.to_tokens(self.tokenizer.encode)[0][1:], out[0, :-1, :]
                        )
                    print(txt, end="")
                    sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if token_out:
                        pass
                    elif stream:
                        package = {"next": txt}
                        if need_ppl:
                            package["in_ppls"] = in_ppls
                            package["out_ppls"] = out_ppls
                        yield json.dumps(package, ensure_ascii=True) + "\n"
                    else:
                        if need_ppl:
                            yield txt, in_ppls, out_ppls
                        else:
                            yield txt
                elif block["type"] == "select":
                    selections = block["selections"]
                    select_tokens = [
                        self.tokenizer.encode(s.strip())
                        for s in selections
                        if s.strip()
                    ]
                    while select_tokens:
                        allow_tokens = [s[0] for s in select_tokens]
                        for n in occurrence:
                            out[0, -1, n] -= (
                                presence_penalty + occurrence[n] * frequency_penalty
                            )
                        if allow_tokens:
                            mask = torch.full_like(
                                out[0, -1, :], -1e38, device=out.device
                            )
                            mask[allow_tokens] = out[0, -1, allow_tokens]
                            out[0, -1, :] = mask
                        for tttt in token_ban:
                            out[0, -1, tttt] = torch.full_like(
                                out[0, -1, tttt], -1e38, device=out.device
                            )
                        token = sample_logits(
                            out[0, -1, :],
                            temperature=temp,
                            top_p=top_p,
                        )
                        if need_ppl:
                            token_ppl = ppl([token], out[0, -1:, :])
                            out_ppls += token_ppl
                        for xxx in occurrence:
                            occurrence[xxx] *= decay_penalty
                        if 49 <= token <= 58:
                            pass
                        elif token not in occurrence:
                            occurrence[token] = 1
                        else:
                            occurrence[token] += 1
                        select_tokens = [
                            s[1:]
                            for s in select_tokens
                            if s and token == s[0] and s[1:]
                        ]
                        # print("select->" ,select_tokens)
                        out, new_state = self.model.infer([token], new_state)
                        tokens += [token]
                        if token_out:
                            yield token
                        count += 1
                        if count > stream_chunk and "�" not in self.tokenizer.decode(
                            tokens[-1:]
                        ):
                            count = 0
                            txt = self.tokenizer.decode(tokens[start_index:])
                            start_index = len(tokens)
                            if token_out:
                                pass
                            elif stream:
                                package = {"next": txt}
                                if need_ppl:
                                    package["in_ppls"] = in_ppls
                                    package["out_ppls"] = out_ppls
                                yield json.dumps(package, ensure_ascii=True) + "\n"
                            # else:
                            print(txt, end="")
                            sys.stdout.flush()
                    if count > 0:
                        count = 0
                        txt = self.tokenizer.decode(tokens[start_index:])
                        start_index = len(tokens)
                        if token_out:
                            pass
                        elif stream:
                            package = {"next": txt}
                            if need_ppl:
                                package["in_ppls"] = in_ppls
                                package["out_ppls"] = out_ppls
                            yield json.dumps(package, ensure_ascii=True) + "\n"
                        # else:
                        print(txt, end="")
                        sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if not stream:
                        send_msg = self.tokenizer.decode(tokens)
                        tokens = []
                        if need_ppl:
                            yield send_msg, in_ppls, out_ppls
                        elif not token_out:
                            yield send_msg
                elif block["type"] == "generate":
                    max_len = block["n_max"]
                    stop = block["stop"]
                    supplement = block["supplement"]
                    for i in range(max_len):
                        for n in occurrence:
                            out[0, -1, n] -= (
                                presence_penalty + occurrence[n] * frequency_penalty
                            )
                        for tttt in token_ban:
                            out[0, -1, tttt] = torch.full_like(
                                out[0, -1, tttt], -1e38, device=out.device
                            )

                        token = sample_logits(
                            out[0, -1, :],
                            temperature=temp,
                            top_p=top_p,
                        )
                        if need_ppl:
                            token_ppl = ppl([token], out[0, -1:, :])
                            out_ppls += token_ppl
                        for xxx in occurrence:
                            occurrence[xxx] *= decay_penalty
                        if 49 <= token <= 58:
                            pass
                        elif token not in occurrence:
                            occurrence[token] = 1
                        else:
                            occurrence[token] += 1
                        out, new_state = self.model.infer([token], new_state)
                        tokens += [token]
                        count += 1
                        if token in stop or i == max_len - 1:  # 结束
                            if supplement:
                                out, new_state = self.model.infer(supplement, new_state)
                                tokens.pop()
                                count -= 1
                            if count > 0:
                                count = 0
                                txt = self.tokenizer.decode(tokens[start_index:])
                                start_index = len(tokens)
                                if token_out:
                                    pass
                                elif stream:
                                    package = {"next": txt}
                                    if need_ppl:
                                        package["in_ppls"] = in_ppls
                                        package["out_ppls"] = out_ppls
                                    yield json.dumps(package, ensure_ascii=True) + "\n"
                                # else:
                                print(txt, end="")
                                sys.stdout.flush()
                            break
                        if token_out:
                            yield token
                        if count > stream_chunk and "�" not in self.tokenizer.decode(
                            tokens[-1:]
                        ):
                            count = 0
                            txt = self.tokenizer.decode(tokens[start_index:])
                            start_index = len(tokens)
                            if token_out:
                                pass
                            elif stream:
                                package = {"next": txt}
                                if need_ppl:
                                    package["in_ppls"] = in_ppls
                                    package["out_ppls"] = out_ppls
                                yield json.dumps(package, ensure_ascii=True) + "\n"
                            # else:
                            print(txt, end="")
                            sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if not stream:
                        send_msg = self.tokenizer.decode(tokens)
                        tokens = []
                        if need_ppl:
                            yield send_msg, in_ppls, out_ppls
                        elif not token_out:
                            yield send_msg
        else:
            for i in range(max_resp_len):
                for n in occurrence:
                    out[0, -1, n] -= (
                        presence_penalty + occurrence[n] * frequency_penalty
                    )
                for tttt in token_ban:
                    out[0, -1, tttt] = torch.full_like(
                        out[0, -1, tttt], -1e38, device=out.device
                    )

                token = sample_logits(
                    out[0, -1, :],
                    temperature=temp,
                    top_p=top_p,
                )
                if need_ppl:
                    token_ppl = ppl([token], out[0, -1:, :])
                    out_ppls += token_ppl
                for xxx in occurrence:
                    occurrence[xxx] *= decay_penalty
                if 49 <= token <= 58:
                    pass
                elif token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1
                out, new_state = self.model.infer([token], new_state)
                tokens += [token]
                count += 1
                if token in stop_with_tokens or i == max_resp_len - 1:  # 结束
                    tokens.pop()
                    count -= 1
                    if stop_supp_tokens:
                        out, new_state = self.model.infer(stop_supp_tokens, new_state)
                    if count > 0:
                        count = 0
                        txt = self.tokenizer.decode(tokens[start_index:])
                        start_index = len(tokens)
                        if token_out:
                            pass
                        elif stream:
                            package = {"next": txt}
                            if need_ppl:
                                package["in_ppls"] = in_ppls
                                package["out_ppls"] = out_ppls
                            yield json.dumps(package, ensure_ascii=True) + "\n"
                        # else:
                        print(txt, end="")
                        sys.stdout.flush()
                    break
                if token_out:
                    yield token
                if count > stream_chunk and "�" not in self.tokenizer.decode(
                    tokens[-1:]
                ):
                    count = 0
                    txt = self.tokenizer.decode(tokens[start_index:])
                    start_index = len(tokens)
                    if token_out:
                        pass
                    elif stream:
                        package = {"next": txt}
                        if need_ppl:
                            package["in_ppls"] = in_ppls
                            package["out_ppls"] = out_ppls
                        yield json.dumps(package, ensure_ascii=True) + "\n"
                    # else:
                    print(txt, end="")
                    sys.stdout.flush()
            if save_to_now_state_idx:
                self.states_pool[save_to_now_state_idx] = new_state
            if not stream:
                send_msg = self.tokenizer.decode(tokens)
                tokens = []
                if need_ppl:
                    yield send_msg, out_ppls
                elif not token_out:
                    yield send_msg

    def batch_chat(
        self,
        start_with_tokens_batch: List[List[int]],
        stop_with_tokens: List[int],
        stop_supp_tokens: List[int],
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        decay_penalty: float,
        use_now_state_idx_batch=None,
        save_to_now_state_idx_batch=None,
        max_resp_len: int = 512,
        token_ban=[],
    ):
        """
        效率有待改进，暂时用这个
        """
        B = len(start_with_tokens_batch)

        # 组装states
        if use_now_state_idx_batch is None:
            states = RWKVStates.create(
                self.model.args.n_layer,
                B,
                self.model.args.n_embd,
                self.model.args.n_head,
                self.model.args.head_size,
                self.model.device,
                self.model.dtype,
            )
        else:
            assert len(use_now_state_idx_batch) == B
            states = []
            for i in use_now_state_idx_batch:
                state = self.states_pool[i]
                if state is None:
                    state = RWKVStates.create(
                        self.model.args.n_layer,
                        1,
                        self.model.args.n_embd,
                        self.model.args.n_head,
                        self.model.args.head_size,
                        next(self.model.parameters()).device,
                        next(self.model.parameters()).dtype,
                    )
                states.append(state)
            states = sum(states[1:], states[0])

        speak_sequences_batch, speak_texts_batch, out_states = batch_chat(
            rwkv=self.model,
            start_with_tokens_batch=start_with_tokens_batch,
            tokenizer=self.tokenizer,
            stop_with_tokens=stop_with_tokens,
            stop_supp_tokens=stop_supp_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            decay_penalty=decay_penalty,
            batch_state=states,
            max_resp_len=max_resp_len,
            token_ban=token_ban,
        )

        out_states = out_states.unbind()
        if save_to_now_state_idx_batch is not None:
            for i, state in enumerate(out_states):
                self.states_pool[save_to_now_state_idx_batch[i]] = state

        return speak_sequences_batch, speak_texts_batch
    
    def run_chat_task(
        self,
        start_with_tokens: List[int],
        stop_with_tokens: List[int],
        stop_supp_tokens: List[int],
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        penalty_decay: float,
        constaraint_str: str,
        use_now_state_idx=None,
        save_to_now_state_idx=None,
        max_resp_len: int = 512,
        token_ban=[0],
        occurence={},
        stream_chunk=9,
    ):
        start_with_states = (
            self.states_pool[use_now_state_idx]
            if use_now_state_idx
            else None
        )
        buffer_tokens=[]
        n_tokens=0
        for resp_tokens, state,in_ppls, out_ppls, over in self.batching_manager.run_task(
            begin_with_tokens=start_with_tokens,
            state=start_with_states,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            constraint_str=constaraint_str,
            token_stop=stop_with_tokens,
            token_stop_supp=stop_supp_tokens,
            max_tokens=max_resp_len,
            token_ban=token_ban,
            occurence=occurence,
        ):
            n_tokens+=len(resp_tokens)
            if n_tokens>max_resp_len:
                break
            if resp_tokens:
                buffer_tokens.extend(resp_tokens)
                # 尝试 decode 累积的 tokens
                resp_txt = self.tokenizer.decode(buffer_tokens)
                if "�" in resp_txt or len(buffer_tokens) < stream_chunk:
                    continue
                else:
                    print(resp_txt, end="", flush=True)
                    package = {
                        "next": resp_txt,
                        "in_ppls": in_ppls,
                        "out_ppls": out_ppls,
                    }
                    yield json.dumps(package, ensure_ascii=False) + "\n"
                    buffer_tokens.clear()
        if buffer_tokens:
            resp_txt = self.tokenizer.decode(buffer_tokens)
            print(resp_txt, end="", flush=True)
            package = {
                "next": resp_txt,
                "in_ppls": in_ppls,
                "out_ppls": out_ppls,
            }
            yield json.dumps(package, ensure_ascii=False) + "\n"
            buffer_tokens.clear()
            
            
            
            
        if save_to_now_state_idx:
            self.states_pool[save_to_now_state_idx] = state


    def estimate_desires(
        self,
        target_tokens_list: List[List[int]],
        start_with_tokens: List[int],
        ignore_tokens=[11, 33, 261, 263, 41, 42],
        ignore_tolerance=2,
        use_now_state_idx=None,
    ):
        max_len = (
            max([len(target_tokens) for target_tokens in target_tokens_list])
            + ignore_tolerance
        )
        resp_tokens = []
        for t in self.chat(
            None,
            start_with_tokens,
            [],
            [],
            0.2,
            1,
            0,
            0,
            1,
            use_now_state_idx,
            max_resp_len=max_len,
            token_out=True,
        ):
            resp_tokens.append(t)
        for t in ignore_tokens:
            while t in resp_tokens:
                resp_tokens.remove(t)
            for target_tokens in target_tokens_list:
                while t in target_tokens:
                    target_tokens.remove(t)

        match_index = -1
        print(f"start_with_tokens: {start_with_tokens}")
        for i, target_tokens in enumerate(target_tokens_list):
            print(f"estimate desire {i}=>{resp_tokens}←→{target_tokens}")
            if resp_tokens[: len(target_tokens)] == target_tokens:
                match_index = i
                break
        return match_index
