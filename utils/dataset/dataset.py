from config import global_config
import os
from collections import OrderedDict
import pyarrow.parquet as pq
import pandas as pd

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


from torch.utils.data import Dataset
import json
import torch, torchaudio
import os
from utils.message_manager import cList, Conversation
import random
from datetime import datetime
import tqdm
from concurrent.futures import ThreadPoolExecutor

# 扫描某个文件夹，取得文件夹下的所有jsonl、MP4等格式文件
# 然后记录每个文件的ctx到文件
# 每次随机截取ctx_len的数据


voice_sos_tokens = global_config.role["voice"]["prefix"]
voice_eos_tokens = global_config.role["voice"]["postfix"]


class MultimodalDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        voice_read_func,
        video_load_func,
        ctx_len,
        qa_mask_on=False,
    ):
        """
        --dataset_dir
          --children_folder1
          --children_folder2
            --text_data1.txt
            --text_data2.jsonl
            --text_voice_data.jsonl
            --voice_picture_data.mp4

        MultimodalDataset使用“units”作为基本单位。
        units是一个列表，其中的元素unit对应ctx_len为1时的一个单位不同模态数据。
        例如，文本的unit是一个int token
        音频的unit是一个torch.Tensor(1,2,voice_unit_len)长度的双声道音频
        文本和音频的unit是混合在一个列表中的。代表“潜空间同表示”
        视频等其他模态则单独出一个列表，代表“嵌入”
        """
        super().__init__()
        self.eod = 0
        self.dataset_dir = dataset_dir
        self.idx_dir = f"{dataset_dir}/idx.index"
        self.file_level_idx_folder = f"{dataset_dir}/in_line_indices"
        self.dataset_metadata_dir = f"{dataset_dir}/dataset_metadata.json"
        os.makedirs(self.file_level_idx_folder, exist_ok=True)
        self.tokenize_func = tokenizer.encode
        self.detokenize_func = tokenizer.decode
        self.voice_read_func = voice_read_func
        self.video_load_func = video_load_func
        self.ctx_len = ctx_len
        self.qa_mask_on = qa_mask_on
        if not os.path.exists(self.idx_dir) or not os.path.exists(
            self.dataset_metadata_dir
        ):
            self.calc_and_save_idx_file()
            self.read_metadatas()
        else:
            self.read_metadatas()

    def read_metadatas(self):
        with open(self.dataset_metadata_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.total_ctx = data["total_ctx"]

    def calc_and_save_idx_file(self):
        """
        计算并保存数据集ctx记录txt
        """
        print("indexing...")
        acc_ctx = 0
        idx_str = ""
        for root, dirs, files in os.walk(self.dataset_dir):
            date_time = datetime.now()
            fmt = "%Y-%m-%d %H:%M:%S"
            dt_str = date_time.strftime(fmt)
            loop = tqdm.tqdm(files)
            for file in loop:
                fp = os.path.join(root, file)
                loop.set_postfix(processing=fp)
                units, masks = self.encode_fp_to_units(fp, True)
                if units:
                    ctx_len = len(units)
                    idx_str += f"{fp}|{acc_ctx}|{ctx_len}|{dt_str}\n"
                    acc_ctx += ctx_len

        with open(self.idx_dir, "w", encoding="utf-8") as f:
            f.write(idx_str.rstrip("\n"))
        with open(self.dataset_metadata_dir, "w", encoding="utf-8") as f:
            f.write(json.dumps({"total_ctx": acc_ctx}))
        return acc_ctx

    def process_jsonl_dataset_line(self, line):
        """
        jsonl数据分为两类，一类是rwkv_default数据，如
        {"text": datas}
        另外一类是sp_token数据，如：
        {
            "data": [{"role":xxx, "content":xxx, voice:"xxx"}]
        }

        联合编码文本和音频，也可能只有文本
        返回一个列表，文本会被编码成token(int)；音频会被读取为(1, 1, chunk_len_per_token)的torch.tensor
        """
        line_units = []
        line_masks = []
        line_dict = json.loads(line)
        # V2 dataset
        if "data_dict_list" in line_dict:
            line_clist = cList.from_dicts(line_dict["data_dict_list"])
            for conversation in line_clist:
                last_len = len(line_units)
                sos_tokens = global_config.role[conversation.role]["prefix"]
                eos_tokens = global_config.role[conversation.role]["postfix"]

                line_units += sos_tokens
                c_str = conversation()
                n_voice = -1
                while c_str.find("<-voice->") != -1:
                    n_voice += 1
                    voice_index = c_str.find("<-voice->")
                    pre_str = c_str[:voice_index]
                    line_units += self.tokenize_func(pre_str) + voice_sos_tokens
                    voice_unit_len = (
                        config.vocoder.head.hop_length
                        * config.vocoder.adapter.chunk_len
                    )
                    line_units += (
                        list(
                            torch.split(
                                self.voice_read_func(
                                    conversation.voice[n_voice], config
                                ).unsqueeze(0),
                                voice_unit_len,
                                dim=2,
                            )
                        )
                        + voice_eos_tokens
                    )
                    c_str = c_str.replace("<-voice->", "", 1)
                    c_str = c_str[voice_index:]
                line_units += self.tokenize_func(c_str) + eos_tokens

                prefix_masks = [conversation.prefix_mask] * len(sos_tokens)
                content_masks = [conversation.content_mask] * (
                    len(line_units) - last_len - len(sos_tokens) - len(eos_tokens)
                )
                post_masks = [conversation.postfix_mask] * len(eos_tokens)
                qa_mask = (
                    int(conversation.role in global_config.ego_types)
                    if self.qa_mask_on
                    else 1
                )

                line_masks += [
                    ele * qa_mask for ele in prefix_masks + content_masks + post_masks
                ]

            line_units += [self.eod]
            line_masks += [0]
        # V1 dataset
        elif "data" in line_dict:
            line_clist = cList.from_v1_datset_dicts(line_dict["data"])
            for conversation in line_clist:
                last_len = len(line_units)
                sos_tokens = global_config.role[conversation.role]["prefix"]
                eos_tokens = global_config.role[conversation.role]["postfix"]

                line_units += sos_tokens
                c_str = conversation()
                line_units += self.tokenize_func(c_str) + eos_tokens

                qa_mask = (
                    int(conversation.role in global_config.ego_types)
                    if self.qa_mask_on
                    else 1
                )
                line_masks += [1 * qa_mask] * (len(line_units) - last_len)

            line_units += [self.eod]
            line_masks += [0]
        # rwkv dataset
        elif "text" in line_dict:
            line_units = self.tokenize_func(line_dict)
            line_masks += [1] * len(line_units)
            line_units += [self.eod]
            line_masks += [0]

        else:
            pass
        assert len(line_units) == len(line_masks)
        return line_units, line_masks

    def encode_fp_to_units(self, fp: str, make_file_level_idx: bool = False):
        units = []
        masks = []
        if fp.lower().endswith(".jsonl"):
            idx_fn = fp.replace("/", "-")
            idx_fp = f"{self.file_level_idx_folder}/{idx_fn}.idx"
            file_level_idx = ""
            now_ctx = 0
            with open(fp, "r", encoding="utf-8") as file:
                for i, line in enumerate(file, 0):
                    line_units, line_masks = self.process_jsonl_dataset_line(line)
                    line_ctx_len = len(line_units)
                    if line_ctx_len:
                        file_level_idx += f"{i}|{now_ctx}|{line_ctx_len}\n"
                        now_ctx += line_ctx_len
                        units += line_units
                        masks += line_masks
            if make_file_level_idx:
                with open(idx_fp, "w", encoding="utf-8") as f:
                    f.write(file_level_idx)
        elif fp.lower().endswith(".txt"):
            idx_fn = fp.replace("/", "-")
            idx_fp = f"{self.file_level_idx_folder}/{idx_fn}.idx"
            file_level_idx = ""

            with open(fp, "r", encoding="utf-8") as file:
                content = file.read()
                units = self.tokenize_func(content)
                masks = [1] * len(units)

            if (
                make_file_level_idx and len(units) + 1 > 10000
            ):  # +1 因为 EOD (0 token) 会添加
                bucket_ctx_len = 10000
                unit_start_ctx = 0

                while unit_start_ctx < len(units):
                    bucket_units = units[
                        unit_start_ctx : unit_start_ctx + bucket_ctx_len
                    ]
                    ctx_len = len(bucket_units)
                    bucket_str = self.detokenize_func(
                        bucket_units
                    )  # 将 token 转换回字符串

                    # 确定 bucket_str 在 content 中的起始位置和长度
                    bucket_start_pos = content.find(bucket_str)
                    if bucket_start_pos == -1:
                        raise ValueError("Unable to locate bucket string in content.")
                    bucket_len = len(bucket_str)

                    # 构造索引行
                    file_level_idx += (
                        f"{bucket_start_pos}|{bucket_len}|{unit_start_ctx}|{ctx_len}\n"
                    )

                    # 更新起始位置
                    unit_start_ctx += bucket_ctx_len

                # 将索引写入索引文件
                with open(idx_fp, "w", encoding="utf-8") as f:
                    f.write(file_level_idx)
            units += [self.eod]
            masks += [0]
        assert len(units) == len(masks)
        return units, masks

    def process_txt_dataset(self, txt_fp):
        with open(txt_fp, "r", encoding="utf-8") as file:
            content = file.read()
            units = self.tokenize_func(content) + [self.eod]
        return units

    def process_mp4_dataset(self, mp4):
        raise NotImplementedError
        ctx_len = 0
        return ctx_len

    def __getitem__(self, index=None, ctx_len=None):
        """
        随机截取ctx_len+1长度的units
        """
        start_offset, find_files, end_offset, file_ctxs = self.find_target_file(
            index, ctx_len
        )

        def process_file(args):
            i, (fp, f_ctx) = args
            if 0 < i < len(find_files) - 1:
                return self.encode_fp_to_units(fp)
            else:
                start = 0 if i != 0 else start_offset
                end = f_ctx if i != len(find_files) - 1 else end_offset
                idx_fn = fp.replace("/", "-")
                idx_fp = f"{self.file_level_idx_folder}/{idx_fn}.idx"
                if os.path.exists(idx_fp):
                    return self.get_ctx_by_inline_idx(fp, idx_fp, start, end - start)
                else:
                    f_units, f_masks = self.encode_fp_to_units(fp)
                    return f_units[start:end], f_masks[start:end]

        # 创建参数列表
        args = list(enumerate(zip(find_files, file_ctxs)))

        # 使用线程池并行处理
        units = []
        global_masks = []
        with ThreadPoolExecutor() as executor:
            results = tuple(executor.map(process_file, args))

        # 合并结果
        for result in results:
            units.extend(list(result[0]))
            global_masks.extend(list(result[1]))

        return units, global_masks

    def find_target_file(self, index=None, ctx_len=None):
        """
        随机选择(0, total_ctx-req_len)之间的一个值，记作i，目的是取出ctx在[i,i+req_len]之间的数据
        打开self.idx_dir，将指针挪到中间，查看当前行（格式：文件地址|起始ctx|ctx长度|数据加入时间）中的起始ctx，并使用折半查找锁定当前起始ctx<i，下一行起始ctx>i的行
        根据锁定的行的起始ctx，计算读取当前文件的起始位置，然后看ctx长度是否>req_len，如果是，则需要读取下一行的文件，然后在看下一行文件的ctx长度是否不足剩余长度，来判断是否要继续读取下一个文件。
        最后，按顺序得到一个需要读取的文件列表，第一个文件的起始位置，以及最后一个文件的终止位置。
        """
        req_len = self.ctx_len + 1 if ctx_len is None else ctx_len + 1
        # 随机选择起始位置
        if index == None:
            i = random.randint(0, self.total_ctx - req_len)
        else:
            i = index

        # 二分查找定位文件
        with open(self.idx_dir, "r", encoding="utf-8") as f:
            # 移到文件中间
            f.seek(0, 2)  # 移到文件末尾
            file_size = f.tell()
            left, right = 0, file_size

            while left < right:
                mid = (left + right) // 2
                if mid != 0:
                    f.seek(mid)
                    f.readline()
                    line = f.readline()
                else:
                    f.seek(mid)
                    line = f.readline()
                if not line.strip():
                    "dataset: encountered empty line"
                    right = mid
                    mid = (left + right) // 2
                    continue
                filepath, start_ctx, file_ctx, dt_str = line.strip().split("|")
                start_ctx = int(start_ctx)
                file_ctx = int(file_ctx)

                is_find = False
                find_files = []
                file_ctxs = []
                acc_ctx = 0
                if start_ctx <= i and start_ctx + file_ctx > i:
                    while req_len > acc_ctx:
                        find_files.append(filepath)
                        file_ctxs.append(file_ctx)

                        if not is_find:
                            start_offset = i - start_ctx
                            is_find = True
                            acc_ctx += file_ctx - start_offset
                        else:
                            acc_ctx += file_ctx
                        if req_len <= acc_ctx:  # 目标长度<= 下次积累长度
                            end_offset = req_len - acc_ctx + file_ctx
                            break
                        next_line = f.readline()
                        # 在遍历训练器中，这里存在问题，因为找到了最后一行，而它没有内容
                        filepath, start_ctx, file_ctx, dt_str = next_line.strip().split(
                            "|"
                        )
                        start_ctx = int(start_ctx)
                        file_ctx = int(file_ctx)
                    return start_offset, find_files, end_offset, file_ctxs

                if start_ctx > i:
                    right = mid
                else:
                    left = mid + 1

    def get_ctx_by_inline_idx(
        self, file_path: str, idx_file_path: str, i: int, req_len: int
    ):
        if file_path.lower().endswith(".jsonl"):
            with open(idx_file_path, "r", encoding="utf-8") as index_f, open(
                file_path, "r", encoding="utf-8"
            ) as data_f:
                data_lines = data_f.readlines()
                # 移到文件中间
                index_f.seek(0, 2)  # 移到文件末尾
                file_size = index_f.tell()
                left, right = 0, file_size

                while left < right:
                    mid = (left + right) // 2
                    if mid != 0:
                        index_f.seek(mid)
                        index_f.readline()
                        line = index_f.readline()
                    else:
                        index_f.seek(mid)
                        line = index_f.readline()
                    if not line.strip():
                        "dataset: encountered empty index line"
                        right = mid
                        mid = (left + right) // 2
                        continue

                    line_id, start_ctx, line_ctx = line.strip().split("|")
                    start_ctx = int(start_ctx)
                    line_ctx = int(line_ctx)
                    line_id = int(line_id)
                    is_find = False
                    find_lines = []
                    acc_ctx = 0
                    if start_ctx <= i and start_ctx + line_ctx > i:
                        while req_len > acc_ctx:
                            find_lines.append(line_id)
                            if not is_find:
                                start_offset = i - start_ctx
                                is_find = True
                                acc_ctx += line_ctx - start_offset
                            else:
                                acc_ctx += line_ctx
                            if req_len <= acc_ctx:  # 目标长度<= 下次积累长度
                                end_offset = req_len - acc_ctx + line_ctx
                                break
                            next_line = index_f.readline()
                            line_id, start_ctx, line_ctx = next_line.strip().split("|")
                            start_ctx = int(start_ctx)
                            line_ctx = int(line_ctx)
                            line_id = int(line_id)
                            start_ctx = int(start_ctx)
                            line_ctx = int(line_ctx)
                        # 根据find_lines读取文件
                        units = []
                        masks = []
                        for j, line in enumerate(find_lines, 0):
                            start = 0 if j != 0 else start_offset
                            end = None if j != len(find_lines) - 1 else end_offset
                            data_line = data_lines[line]
                            line_units, line_masks = self.process_jsonl_dataset_line(
                                data_line
                            )
                            units += line_units[start:end]
                            masks += line_masks[start:end]

                        return units, masks

                    if start_ctx > i:
                        right = mid
                    else:
                        left = mid + 1
        elif file_path.lower().endswith(".txt"):
            with open(idx_file_path, "r", encoding="utf-8") as index_f, open(
                file_path, "r", encoding="utf-8"
            ) as data_f:
                # 移到文件中间
                index_f.seek(0, 2)  # 移到文件末尾
                file_size = index_f.tell()
                left, right = 0, file_size

                while left < right:
                    mid = (left + right) // 2
                    if mid != 0:
                        index_f.seek(mid)
                        index_f.readline()
                        line = index_f.readline()
                    else:
                        index_f.seek(mid)
                        line = index_f.readline()
                    if not line.strip():
                        "dataset: encountered empty index line"
                        right = mid
                        mid = (left + right) // 2
                        continue

                    bucket_start_pos, bucket_len, unit_start_ctx, bucket_ctx_len = (
                        line.strip().split("|")
                    )
                    bucket_start_pos = int(bucket_start_pos)
                    bucket_len = int(bucket_len)
                    unit_start_ctx = int(unit_start_ctx)
                    bucket_ctx_len = int(bucket_ctx_len)
                    is_find = False
                    find_idxs = []
                    acc_ctx = 0
                    if unit_start_ctx <= i and unit_start_ctx + bucket_ctx_len > i:
                        while req_len > acc_ctx:
                            find_idxs.append((bucket_start_pos, bucket_len))
                            if not is_find:
                                start_offset = i - unit_start_ctx
                                is_find = True
                                acc_ctx += bucket_ctx_len - start_offset
                            else:
                                acc_ctx += bucket_ctx_len
                            if req_len <= acc_ctx:  # 目标长度<= 下次积累长度
                                end_offset = req_len - acc_ctx + bucket_ctx_len
                                break
                            next_line = index_f.readline()
                            (
                                bucket_start_pos,
                                bucket_len,
                                unit_start_ctx,
                                bucket_ctx_len,
                            ) = next_line.strip().split("|")
                            bucket_start_pos = int(bucket_start_pos)
                            bucket_len = int(bucket_len)
                            unit_start_ctx = int(unit_start_ctx)
                            bucket_ctx_len = int(bucket_ctx_len)
                        # 根据find_lines读取文件
                        units = []
                        masks = []
                        for j, (bucket_start_pos, bucket_len) in enumerate(
                            find_idxs, 0
                        ):
                            start = 0 if j != 0 else start_offset
                            end = None if j != len(find_idxs) - 1 else end_offset
                            data_f.seek(bucket_start_pos)
                            content = data_f.read(bucket_len)
                            units += self.tokenize_func(content)[start:end]
                            masks += [1] * len(units)
                        return units, masks

                    if unit_start_ctx > i:
                        right = mid
                    else:
                        left = mid + 1

    def add_data(self, fp: str, date_time=None):
        """
        计算并保存数据集ctx记录txt
        """
        if date_time is None:
            date_time = datetime.now()
            fmt = "%Y-%m-%d %H:%M:%S"
            dt_str = date_time.strftime(fmt)
        acc_ctx = self.total_ctx
        units, masks = self.encode_fp_to_units(fp)
        ctx_len = len(units)
        add_str = f"{fp}|{acc_ctx}|{ctx_len}|{dt_str}\n"
        acc_ctx += ctx_len
        with open(self.idx_dir, "a", encoding="utf-8") as f:
            f.write(add_str)
        self.total_ctx = acc_ctx
        with open(self.dataset_metadata_dir, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["total_ctx"] = acc_ctx
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4, ensure_ascii=False)
        return acc_ctx


@torch.no_grad()
def read_wav(fp, config):
    y, sr = torchaudio.load(fp)
    # if y.size(0) > 1:
    #     # mix to mono
    #     y = y.mean(dim=0, keepdim=True)
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
            config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
            - last_length,
        )
        y = torch.cat((y, padding_tensor), dim=-1)
    return y


@torch.no_grad()
def read_bin_wav(fp, config):
    wav = read_wav(fp, config)
    ch, N = wav.size()
    if ch == 1:
        return torch.cat([wav.clone(), wav.clone()], dim=0)
    else:
        return wav


class RLGroupDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer,
        voice_read_func: callable,
        qa_mask_on: bool = True,
    ):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.voice_read_func = voice_read_func
        self.qa_mask_on = qa_mask_on
        self.datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")]
        self.eod = 0

    @staticmethod
    def read_jsonl_as_ordered_dicts(file_path):
        ordered_dicts = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # 将每一行解析为有序字典
                ordered_dict = json.loads(line, object_pairs_hook=OrderedDict)
                ordered_dicts.append(ordered_dict)
        return ordered_dicts

    def encode_conversations_to_units(self, conversations: cList):
        line_units = []
        line_masks = []
        for conversation in conversations:
            last_len = len(line_units)
            sos_tokens = global_config.role[conversation.role]["prefix"]
            eos_tokens = global_config.role[conversation.role]["postfix"]

            line_units += sos_tokens
            c_str = conversation()
            n_voice = -1
            while c_str.find("<-voice->") != -1:
                n_voice += 1
                voice_index = c_str.find("<-voice->")
                pre_str = c_str[:voice_index]
                line_units += self.tokenizer.encode(pre_str) + voice_sos_tokens
                voice_unit_len = (
                    config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
                )
                line_units += (
                    list(
                        torch.split(
                            self.voice_read_func(
                                conversation.voice[n_voice], config
                            ).unsqueeze(0),
                            voice_unit_len,
                            dim=2,
                        )
                    )
                    + voice_eos_tokens
                )
                c_str = c_str.replace("<-voice->", "", 1)
                c_str = c_str[voice_index:]
            line_units += self.tokenizer.encode(c_str) + eos_tokens

            prefix_masks = [conversation.prefix_mask] * len(sos_tokens)
            content_masks = [conversation.content_mask] * (
                len(line_units) - last_len - len(sos_tokens) - len(eos_tokens)
            )
            post_masks = [conversation.postfix_mask] * len(eos_tokens)
            qa_mask = (
                int(conversation.role in global_config.ego_types)
                if self.qa_mask_on
                else 1
            )

            line_masks += [
                ele * qa_mask for ele in prefix_masks + content_masks + post_masks
            ]

        line_units += [self.eod]
        line_masks += [0]
        return line_units, line_masks

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):

        rl_dict = self.read_jsonl_as_ordered_dicts(
            f"{self.dataset_dir}/{self.datasets[idx]}"
        )

        lines = []

        for line in rl_dict:
            turns = []
            for i, (key, choices) in enumerate(line.items()):
                turn_rollouts = []
                for j, choice in enumerate(choices):
                    score = choice["score"]
                    conversation_dict = choice["choice"]
                    is_best = choice["is_best"]
                    conversations = cList.from_dicts(conversation_dict)
                    units, masks = self.encode_conversations_to_units(conversations)
                    turn_rollouts.append(
                        {
                            "units": units,
                            "masks": masks,
                            "score": score,
                            "is_best": is_best,
                        }
                    )

                turns.append(turn_rollouts)
            lines.append(turns)

        return lines


# class RLGroupDatasetOnline:
#     def __init__(
#         self,
#         history,
#         tokenizer,
#     ):
#         self.history= history
#         # List[List[(cList, float)]]: turns[regenerates[conversations, reawrds]]

#         self.tokenizer = tokenizer

#     def encode_conversations_to_units(self, conversations: cList):
#         line_units = []
#         line_masks = []
#         for conversation in conversations:
#             last_len = len(line_units)
#             sos_tokens = global_config.role[conversation.role]["prefix"]
#             eos_tokens = global_config.role[conversation.role]["postfix"]

#             line_units += sos_tokens
#             c_str = conversation()
#             n_voice = -1
#             while c_str.find("<-voice->") != -1:
#                 n_voice += 1
#                 voice_index = c_str.find("<-voice->")
#                 pre_str = c_str[:voice_index]
#                 line_units += self.tokenizer.encode(pre_str) + voice_sos_tokens
#                 voice_unit_len = (
#                     config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
#                 )
#                 line_units += (
#                     list(
#                         torch.split(
#                             self.voice_read_func(
#                                 conversation.voice[n_voice], config
#                             ).unsqueeze(0),
#                             voice_unit_len,
#                             dim=2,
#                         )
#                     )
#                     + voice_eos_tokens
#                 )
#                 c_str = c_str.replace("<-voice->", "", 1)
#                 c_str = c_str[voice_index:]
#             line_units += self.tokenizer.encode(c_str) + eos_tokens

#             prefix_masks = [conversation.prefix_mask] * len(sos_tokens)
#             content_masks = [conversation.content_mask] * (
#                 len(line_units) - last_len - len(sos_tokens) - len(eos_tokens)
#             )
#             post_masks = [conversation.postfix_mask] * len(eos_tokens)
#             qa_mask = (
#                 int(conversation.role in global_config.ego_types)
#                 if self.qa_mask_on
#                 else 1
#             )

#             line_masks += [
#                 ele * qa_mask for ele in prefix_masks + content_masks + post_masks
#             ]

#         line_units += [self.eod]
#         line_masks += [0]
#         return line_units, line_masks

#     def get_episode_data(self):
#         turns = []
#         for turn in self.history:
#             turn_rollouts = []
#             for rollout in turn:
#                 score = rollout[1]
#                 conversations = rollout[0]
#                 units, masks = self.encode_conversations_to_units(conversations)
#                 turn_rollouts.append(
#                     {
#                         "units": units,
#                         "masks": masks,
#                         "score": score,
#                     }
#                 )
#             turns.append(turn_rollouts)
#         return turns

class SFTDatasetOnline:
    def __init__(self,
                 history,
                 tokenizer,
                 ):
        self.history= history
        self.tokenizer = tokenizer
        self.eod=0
        
    def encode_conversations_to_units(self, conversations: cList):
        line_units = []
        line_masks = []
        for conversation in conversations:
            last_len = len(line_units)
            sos_tokens = global_config.role[conversation.role]["prefix"]
            eos_tokens = global_config.role[conversation.role]["postfix"]
            line_units += sos_tokens
            c_str = conversation()
            n_voice = -1
            while c_str.find("<-voice->")!= -1:
                n_voice += 1
                voice_index = c_str.find("<-voice->")
                pre_str = c_str[:voice_index]
                line_units += self.tokenizer.encode(pre_str) + voice_sos_tokens
                voice_unit_len = (
                    config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
                )
                line_units += (
                    list(
                        torch.split(
                            self.voice_read_func(
                                conversation.voice[n_voice], config
                            ).unsqueeze(0),
                            voice_unit_len,
                            dim=2,
                        )
                    )
                    + voice_eos_tokens
                )
                c_str = c_str.replace("<-voice->", "", 1)
                c_str = c_str[voice_index:]
            line_units += self.tokenizer.encode(c_str) + eos_tokens
            prefix_masks = [conversation.prefix_mask] * len(sos_tokens) 
            content_masks = [conversation.content_mask] * (
                len(line_units) - last_len - len(sos_tokens) - len(eos_tokens)
            )
            post_masks = [conversation.postfix_mask] * len(eos_tokens)
            line_masks += [
                ele for ele in prefix_masks + content_masks + post_masks
            ]
        line_units += [self.eod]
        line_masks += [0]
        return line_units, line_masks
    
    def get_epoch_data(self):
        turns = []
        for turn in self.history:
            conversations=turn[-1]
            units, masks = self.encode_conversations_to_units(conversations)
            turns.append(
                {
                    "units": units,
                    "masks": masks,
                }
            )
        return turns

class RLGroupDatasetOnline:
    def __init__(
        self,
        history,
        tokenizer,
    ):
        self.history = history
        # List[List[(cList, float)]]: turns[regenerates[conversations]]

        self.tokenizer = tokenizer

    def encode_conversations_to_units(self, conversations: cList):
        segment_list = []
        line_units = []
        line_masks = []
        for conversation in conversations:
            last_len = len(line_units)
            sos_tokens = global_config.role[conversation.role]["prefix"]
            eos_tokens = global_config.role[conversation.role]["postfix"]

            line_units += sos_tokens
            c_str = conversation()
            n_voice = -1
            while c_str.find("<-voice->") != -1:
                n_voice += 1
                voice_index = c_str.find("<-voice->")
                pre_str = c_str[:voice_index]
                line_units += self.tokenizer.encode(pre_str) + voice_sos_tokens
                voice_unit_len = (
                    config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
                )
                line_units += (
                    list(
                        torch.split(
                            self.voice_read_func(
                                conversation.voice[n_voice], config
                            ).unsqueeze(0),
                            voice_unit_len,
                            dim=2,
                        )
                    )
                    + voice_eos_tokens
                )
                c_str = c_str.replace("<-voice->", "", 1)
                c_str = c_str[voice_index:]
            line_units += self.tokenizer.encode(c_str) + eos_tokens

            prefix_masks = [conversation.prefix_mask] * len(sos_tokens)
            content_masks = [conversation.content_mask] * (
                len(line_units) - last_len - len(sos_tokens) - len(eos_tokens)
            )
            post_masks = [conversation.postfix_mask] * len(eos_tokens)

            role = conversation.role
            segment_score = conversation.score
            mask_mul = int(
                role in global_config.ego_types and segment_score is not None
            )
            line_masks += [
                ele * mask_mul for ele in prefix_masks + content_masks + post_masks
            ]

            if role in global_config.ego_types and segment_score is not None:
                segment_list.append(
                    {
                        "units": line_units,
                        "masks": line_masks,
                        "score": segment_score,
                    }
                )
                line_masks = [0 for _ in line_masks]

        return segment_list

    def get_episode_data(self):
        turns = []
        for turn in self.history:
            turn_rollouts = []
            for rollout in turn:
                conversations = rollout
                # 需要保留历史信息才行，现在写的没有保留
                segment_list = self.encode_conversations_to_units(conversations)
                turn_rollouts += segment_list
            turns.append(turn_rollouts)
        random.shuffle(turns)
        return turns


class RLPairDataset(Dataset):
    def __init__(
        self,
        dataset_fp: str,
        tokenizer,
        n_samples_episode: int = 5,
        n_episodes=5,
        role_system="system",
        system_prefix="System",
        role_sender="conversation",
        sender_prefix="Q",
        role_replier="response",
        replier_prefix="A",
    ):
        self.tokenizer = tokenizer
        if dataset_fp.endswith(".parquet"):
            self.table = pq.read_table(dataset_fp)
            self.df = self.table.to_pandas()
        elif dataset_fp.endswith(".jsonl"):
            self.df = pd.read_json(dataset_fp, lines=True)
        self.eod = 0
        self.n_samples_episode = n_samples_episode
        self.n_episodes = n_episodes
        self.role_system = role_system
        self.system_prefix = system_prefix
        self.role_sender = role_sender
        self.sender_prefix = sender_prefix
        self.role_replier = role_replier
        self.replier_prefix = replier_prefix

    def __len__(self):
        return len(self.df) // self.n_samples_episode

    def __getitem__(self, idx):
        lines = []
        turns = []
        for i in range(self.n_samples_episode * self.n_episodes):
            row = self.df.iloc[idx + i]
            system_prompt = row.get("system_prompt", None)
            question = row["question"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            chosen_conversations = cList()
            rejected_conversations = cList()
            if system_prompt is not None:
                chosen_conversations.append(
                    Conversation(
                        role=self.role_system,
                        content=f"{self.system_prefix}: {system_prompt}",
                    )
                )
                rejected_conversations.append(
                    Conversation(
                        role=self.role_system,
                        content=f"{self.system_prefix}: {system_prompt}",
                    )
                )
            chosen_conversations.append(
                Conversation(
                    role=self.role_sender,
                    content=f"{self.sender_prefix}: {question}",
                )
            )
            rejected_conversations.append(
                Conversation(
                    role=self.role_sender,
                    content=f"{self.sender_prefix}: {question}",
                )
            )
            chosen_conversations.append(
                Conversation(
                    role=self.role_replier,
                    content=f"{self.replier_prefix}: {chosen}",
                )
            )
            rejected_conversations.append(
                Conversation(
                    role=self.role_replier,
                    content=f"{self.replier_prefix}: {rejected}",
                )
            )

            chosen_tokens, chosen_masks = chosen_conversations.to_tokens(
                self.tokenizer.encode, use_ego_mask=True
            )
            rejected_tokens, rejected_masks = rejected_conversations.to_tokens(
                self.tokenizer.encode, use_ego_mask=True
            )

            turn_rollouts = [
                {
                    "units": chosen_tokens,
                    "masks": chosen_masks,
                    "score": 1,
                    "is_best": True,
                },
                {
                    "units": rejected_tokens,
                    "masks": rejected_masks,
                    "score": -1,
                    "is_best": False,
                },
            ]

            turns.append(turn_rollouts)
            if (i + 1) % self.n_samples_episode == 0:
                lines.append(turns)
                turns = []

        return lines
