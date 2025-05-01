import json
from types import SimpleNamespace
import os
import torch
from rwkv.rwkv_tokenizer import TRIE
from typing import Optional


class FeatureExtractorManager:
    _instance: Optional["FeatureExtractorManager"] = None
    _feature_extractor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 确保只初始化一次
        if FeatureExtractorManager._feature_extractor is None:
            self._load_feature_extractor()

    def _load_feature_extractor(self):
        try:
            FeatureExtractorManager._feature_extractor = torch.load(
                global_config.vocoder.load_feature_extractor_dir
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load feature extractor: {str(e)}")

    @classmethod
    def get_feature_extractor(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._feature_extractor


def singleton_with_id(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        identifier = args[0] if args else None
        if identifier not in instances:
            instances[identifier] = cls(*args, **kwargs)
        return instances[identifier]

    return get_instance


@singleton_with_id
class UNIQUE_TRIE_TOKENIZER:
    def __init__(self, identifier: str, file_name: str):
        self.identifier = identifier
        self.idx2token = {}
        sorted = []  # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode("utf-8")
        except:
            return "\ufffd"  # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()


def dict_to_namespace(d) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def load_config(path, use_default=False) -> SimpleNamespace:
    if use_default:
        with open("./configs/default.json", "r", encoding="utf-8") as f:
            text = f.read()
        config_data = json.loads(text)
        args = dict_to_namespace(config_data)
        return args
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    config_data = json.loads(text)
    args = dict_to_namespace(config_data)
    return args


def namespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}
    elif isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    else:
        return namespace


role = {
    "voice": {"prefix": [6884, 35987, 11], "postfix": [6884, 11]},
    "image": {"prefix": [6884, 34733, 11], "postfix": [6884, 11]},
    "response_voice": {"prefix": [65530, 65528], "postfix": [65535, 11]},
    "search": {"prefix": [65530, 65529], "postfix": [65535, 11]},
    "system": {"prefix": [65530, 65531], "postfix": [65535, 11]},
    "system_no_mask": {"prefix": [65530, 65531], "postfix": [65535, 11]},
    "conversation": {"prefix": [65530, 65532], "postfix": [65535, 11]},
    "conversation_no_mask": {"prefix": [65530, 65532], "postfix": [65535, 11]},
    "think": {"prefix": [65530, 65533], "postfix": [65535, 11]},
    "response": {"prefix": [65530, 65534], "postfix": [65535, 11]},
    "common": {"prefix": [65530], "postfix": [65535, 11]},
    "postfix": {"prefix": [], "postfix": [65535, 11]},
    "text": {"prefix": [], "postfix": []},
    "text_req": {"prefix": [], "postfix": []},
    "rwkv_legacy_eos": {"prefix": [], "postfix": [261]},
    "rwkv_legacy_eos_resp": {"prefix": [], "postfix": [261]},
}
ego_types = [
    "text",
    "rwkv_legacy_eos",
    "think",
    "response",
    "response_voice",
    "system_no_mask",
    "conversation_no_mask",
    "rwkv_legacy_eos_resp",
]


global_config = load_config("./configs/configv7.json", use_default=False)
global_config.role = role  # special token maps
global_config.ego_types = ego_types  # 区分模型输出和其他输入的roles
global_config.tokenizer_train = UNIQUE_TRIE_TOKENIZER(
    "train", global_config.tokenizer_train_dir
)
global_config.tokenizer_eval = UNIQUE_TRIE_TOKENIZER(
    "eval", global_config.tokenizer_eval_dir
)

# if global_config.voice_on:
#     global_config.voice_feat_extractor = FeatureExtractorManager.get_feature_extractor()
if hasattr(global_config.ourborous_config, "deep_reward_model"):
    os.environ["RWKV_REWARD_MODEL_HEAD_SIZE_A"] = str(
        global_config.ourborous_config.deep_reward_model.model.head_size
    )


try:
    working_mode = os.environ["WORKING_MODE"]

    if working_mode == "infer_service":
        os.environ["RWKV_HEAD_SIZE_A"] = str(
            global_config.infer_service_config.model.head_size
        )
        os.environ["RWKV_CTXLEN"] = str(
            global_config.infer_service_config.model.ctx_len
        )
    elif working_mode == "pretrain":
        os.environ["RWKV_HEAD_SIZE_A"] = str(
            global_config.pretrain_script_config.model.head_size
        )
        os.environ["RWKV_CTXLEN"] = str(
            global_config.pretrain_script_config.model.ctx_len
        )
    elif working_mode == "train_service":
        os.environ["RWKV_HEAD_SIZE_A"] = str(
            global_config.train_service_config.model.head_size
        )
        os.environ["RWKV_CTXLEN"] = str(
            global_config.train_service_config.model.ctx_len
        )
    if global_config.rwkv_version == "v6":
        from RWKV.v6.infer_model.model_service import RWKV as RWKVInfer
        from RWKV.v6.rwkv_state.model import RWKV
        # from RWKV.v6.rwkv_state import vocoder
        from RWKV.v6.state import RWKVStates
    if global_config.rwkv_version == "v7":
        os.environ["RWKV_CHUNK_LEN"] = str(
            global_config.train_service_config.model.chunk_len
        )
        from RWKV.v7.infer_model.model_service import RWKV as RWKVInfer
        from RWKV.v7.model import RWKV
        # from RWKV.v7 import vocoder
        from RWKV.v7.state import RWKVStates
except:
    working_mode = "infer_service"
    os.environ["WORKING_MODE"] = "infer_service"
    print(
        "WORKING_MODE not set in environment variables, using infer_service as default."
    )
