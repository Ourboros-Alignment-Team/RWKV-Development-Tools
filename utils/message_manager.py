from config import global_config
import os

config = (
    global_config.infer_service_config
    if os.environ["WORKING_MODE"] == "infer_service"
    else (
        global_config.train_service_config
        if os.environ["WORKING_MODE"] == "train_service"
        else (
            global_config.pretrain_script_config
            if os.environ["WORKING_MODE"] == "pretrain"
            else None
        )
    )
)

from typing import Any
import copy
import os
import torch
import json
import time
from typing import List, Tuple
import random
import torchaudio
import torch.nn.functional as F


@torch.no_grad()
def read_wav(fp):
    y, sr = torchaudio.load(fp)
    if y.size(0) > 1:
        # mix to mono
        y = y.mean(dim=0, keepdim=True)
    gain = -3
    y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
    if sr != config.vocoder.sample_rate:
        y = torchaudio.functional.resample(
            y, orig_freq=sr, new_freq=config.vocoder.sample_rate
        )
    last_length = y.size(-1) % (
        config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
    )
    if last_length != 0:
        padding_tensor = torch.zeros(
            1,
            config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len - last_length,
        )
        y = torch.cat((y, padding_tensor), dim=-1)
    return y


class cList(list):
    @staticmethod
    def from_single_conversation(conversation):
        return cList([conversation])
    
    @classmethod
    def parse_jsonl_to_list_dataset(cls, jsonl_path, key="data", max_line=-1):
        result = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if max_line > 0 and len(result) > max_line:
                    return result
                try:
                    line_dict = json.loads(line)
                    if key in line_dict:
                        dict_list = line_dict[key]
                        line_instance = cls()
                        for d in dict_list:
                            for kk, vv in d.items():
                                conversation = Conversation(role=kk, content=vv)
                                line_instance.append(conversation)
                        result.append(line_instance)
                except json.JSONDecodeError:
                    continue
        return result

    @classmethod
    def parse_jsonl_to_dpo_pair(
        cls, jsonl_path, pos_key="data", neg_key="counterexample", max_line=-1
    ):
        result = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if max_line > 0 and len(result) > line:
                    return result
                try:
                    line_dict = json.loads(line)
                    if pos_key in line_dict:
                        pos_dict_list = line_dict[pos_key]  # dictList
                        neg_dict_list = line_dict[neg_key]
                        pos_line_instance = cls()
                        neg_line_instance = cls()
                        for d in pos_dict_list:
                            for kk, vv in d.items():
                                conversation = Conversation(role=kk, content=vv)
                                pos_line_instance.append(conversation)
                        for d in neg_dict_list:
                            for kk, vv in d.items():
                                conversation = Conversation(role=kk, content=vv)
                                neg_line_instance.append(conversation)
                        result.append((pos_line_instance, neg_line_instance))
                except json.JSONDecodeError:
                    continue
        return result

    @staticmethod
    def from_dicts(data: List[dict]):
        instances = []
        for item in data:
            instance = Conversation.from_dict(item)
            instances.append(instance)
        return cList(instances)

    @staticmethod
    def from_v1_datset_dicts(data: List[dict]):
        instances = []
        for item in data:
            instance = Conversation.from_v1_datset_dict(item)
            instances.append(instance)
        return cList(instances)

    def to_dict_list(self):
        return [item.to_dict() for item in self]

    def to_tokens(
        self, encoding_func: callable, use_ego_mask=False, ego_mask_type="zero"
    ):
        tokens = []
        masks = []
        for c in self:
            tokenc = c.to_tokens(encoding_func, use_ego_mask, ego_mask_type)
            tokens += tokenc[0]
            masks += tokenc[1]
        return tokens, masks

    def __call__(self):
        txt = ""
        for c in self:
            txt += c() + "\n"
        return txt

    def to_dataset(self, data_key="data"):
        dict = {data_key: []}
        txt = ""
        for c in self:
            dict[data_key].append({c.role: c()})
            txt += c() + "\n"
        return dict, txt
    

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, list):
            return cList(result)
        else:
            return result

    def calc_ctx_len(self, encoding_func: callable):
        ctx, _ = self.to_tokens(encoding_func)
        ctx_len = len(ctx)
        return ctx_len

    def __add__(self, other):
        if isinstance(other, cList):
            # 如果 other 也是 cList 类型，将 self 和 other 列表相加，并返回一个新的 cList 实例
            return cList(super().__add__(other))
        else:
            # 如果 other 不是 cList 类型，返回一个普通的 list 相加结果
            return super().__add__(other)
        
    def __str__(self):
        returns="""
```cList object
"""
        for conversation in self:
            returns+=conversation.__str__()
        returns+="""
```
"""
        return returns


class Conversation:
    def __init__(
        self,
        role: str = "",
        content: str = "",
        strip: bool = True,
        prefix_mask: float = 1,
        content_mask: float = 1,
        postfix_mask: float = 1,
        voice: List[str] = [],
        score: float = None,
    ) -> None:
        self.role = role
        self.content = content
        self.strip = strip
        self.prefix_mask = prefix_mask
        self.content_mask = content_mask
        self.postfix_mask = postfix_mask
        self.random_replace_prob = 0.03
        self.voice = voice
        self.score = score
        
    def __call__(self) -> str:
        return self.content if not self.strip else self.content.strip()

    def __str__(self) -> str:
        return_str=f"""conversation:
    role: {self.role}
    content: {self.content}
    strip: {self.strip}
    prefix_mask: {self.prefix_mask}
    content_mask: {self.content_mask}
    postfix_mask: {self.postfix_mask}
    voice: {self.voice}
    score: {self.score}
"""
        return return_str
    

    def to_tokens(
        self,
        encoding_func: callable,
        use_ego_mask=False,
        ego_mask_type="zero",
    ) -> Tuple[List[int], List[int]]:
        assert ego_mask_type in ["zero", "random", "half"]
        sos_tokens = global_config.role[self.role]["prefix"]
        eos_tokens = global_config.role[self.role]["postfix"]
        content_tokens = encoding_func(self())
        prefix_mask = [self.prefix_mask for _ in sos_tokens]
        content_mask = [self.content_mask * int(x != 0) for x in content_tokens]
        if use_ego_mask and self.role not in global_config.ego_types:
            if ego_mask_type == "zero":
                content_mask = [0 for _ in content_tokens]
            elif ego_mask_type == "random":
                content_mask = [random.random() for x in content_mask]
            elif ego_mask_type == "half":
                content_mask = [0.5 * x for x in content_mask]
            for i in range(len(content_tokens)):
                if random.random() < self.random_replace_prob:
                    content_tokens[i] = random.randint(1, 65000)
                    content_mask[i] = 0
        postfix_mask = [self.postfix_mask for _ in eos_tokens]
        return (
            sos_tokens + content_tokens + eos_tokens,
            prefix_mask + content_mask + postfix_mask,
        )

    def to_embds_voice(
        self,
        encoding_func: callable,
        rwkv: callable = None,
        voice_encoder: callable = None,
    ):
        """
        文本-音频混合编码
        """
        sos_tokens = global_config.role[self.role]["prefix"]
        eos_tokens = global_config.role[self.role]["postfix"]
        voice_sos_tokens = global_config.role["voice"]["prefix"]
        voice_eos_tokens = global_config.role["voice"]["postfix"]

        self_str = self()
        prefix_mask = [self.prefix_mask for _ in sos_tokens]
        postfix_mask = [self.postfix_mask for _ in eos_tokens]

        all_tokens = sos_tokens
        sos_tokens = torch.tensor(sos_tokens, dtype=torch.long).unsqueeze(0)
        encoded_embds = rwkv.embedding(sos_tokens).to(
            device=next(rwkv.parameters()).device, dtype=next(rwkv.parameters()).dtype
        )  # B,N,C
        all_mask = prefix_mask
        # print("lens-1",encoded_embds.size(1),len(all_mask),len(all_tokens)) # 这里是正常的

        encoded_voice_index = []
        origin_voice_list = []
        voice_index = 0
        n_voice = -1

        while self_str.find("<-voice->") != -1:
            n_voice += 1
            voice_index = self_str.find("<-voice->")
            # 编码该音频sp_token之前的内容
            pre_str = self_str[:voice_index]
            pre_tokens = encoding_func(pre_str)
            # 增加音频prefix tokens
            pre_tokens += voice_sos_tokens
            pre_content_mask = [self.content_mask * int(x != 0) for x in pre_tokens]
            all_tokens += pre_tokens
            pre_tokens = torch.tensor(pre_tokens, dtype=torch.long).unsqueeze(0)
            pre_embds = rwkv.embedding(pre_tokens)
            # 拼接embds
            encoded_embds = torch.cat((encoded_embds, pre_embds), dim=1)
            all_mask += pre_content_mask
            voice_start_index = encoded_embds.size(1)
            assert encoded_embds.size(1) == len(all_mask)
            # 编码音频的部分
            wav = read_wav(self.voice[n_voice]).unsqueeze(0)
            encoded = voice_encoder(wav)  # 要输入到模型音频端的部分
            with torch.no_grad():
                _, dummy_out_embds = rwkv.forward_without_x(encoded, None)
                # 使用拼接的方法获取第一个参考embds，padding 第一维
                text_embds_with_voice_input = torch.cat(
                    (dummy_out_embds[:, :1], dummy_out_embds[:, :-1]), dim=1
                )
            # 拼接各项
            voice_content_mask = [0 for _ in range(text_embds_with_voice_input.size(1))]
            encoded_embds = torch.cat(
                (encoded_embds, text_embds_with_voice_input), dim=1
            )
            all_tokens += [0 for _ in range(text_embds_with_voice_input.size(1))]
            all_mask += voice_content_mask
            # print("lens1",encoded_embds.size(1),len(all_mask),len(all_tokens))
            assert encoded_embds.size(1) == len(all_mask) == len(all_tokens)
            origin_voice_list.append(wav)
            encoded_voice_index.append(
                (
                    voice_start_index,
                    voice_start_index
                    + encoded.size(2) // config.vocoder.adapter.chunk_len,
                )
            )
            # print("startxxx",voice_start_index,"encoded_size",encoded.size(2)//config.vocoder.adapter.chunk_len)
            # print("xxxxx",encoded_embds.shape,len(all_mask),voice_start_index,voice_start_index+encoded.size(2)//config.vocoder.adapter.chunk_len)
            # 增加音频postfix tokens
            self_str = self_str.replace("<-voice->", "", 1)
            post_tokens = voice_eos_tokens
            post_content_mask = [self.content_mask * int(x != 0) for x in post_tokens]
            post_tokens = torch.tensor(post_tokens, dtype=torch.long).unsqueeze(0)
            post_embds = rwkv.embedding(post_tokens)
            encoded_embds = torch.cat((encoded_embds, post_embds), dim=1)
            all_tokens += voice_eos_tokens
            all_mask += post_content_mask
            self_str = self_str[voice_index:]
            # print("lens1",encoded_embds.size(1),len(all_mask),len(all_tokens))
            assert encoded_embds.size(1) == len(all_mask) == len(all_tokens)

        post_tokens = encoding_func(self_str)
        post_content_mask = [self.content_mask * int(x != 0) for x in post_tokens]
        post_tokens += eos_tokens
        all_tokens += post_tokens
        post_tokens = torch.tensor(post_tokens, dtype=torch.long).unsqueeze(0)
        post_embds = rwkv.embedding(post_tokens)
        encoded_embds = torch.cat((encoded_embds, post_embds), dim=1)
        all_mask += post_content_mask + postfix_mask
        # print("lens2",encoded_embds.size(1),len(all_mask),len(all_tokens))
        assert encoded_embds.size(1) == len(all_mask) == len(all_tokens)

        return (
            encoded_embds,
            all_mask,
            origin_voice_list,
            encoded_voice_index,
            all_tokens,
        )

    @staticmethod
    def from_dict(data: dict):
        instance = Conversation()
        for key, value in data.items():
            setattr(instance, key, value)
        return instance

    @staticmethod
    def from_v1_datset_dict(data: dict):
        instance = Conversation()
        assert len(data) == 1
        instance.role, instance.content = tuple(data.items())[0]
        return instance

    def to_dict(self):
        return {key: getattr(self, key) for key in vars(self)}


class BatchUnit:
    """
    main track -> 多个batch的embds list
    other track -> 各种不同的tensors
    """

    def __init__(self, track_data: list):
        self.tracks = []

    def to_track_inputs(self):
        pass


class BatchUnitList(list):
    """
    默认
    main track -> self.list_tracks[0]
    other track -> self.list_tracks[1:]
    track需要与model的每一个输入接口一一对应，也许可以匹配comfyui
    """

    pass

    def __init__(self, data: List[BatchUnit]):
        self.list_tracks = []
        super().__init__(data)

    def from_clist(self, clist: cList):
        pass
