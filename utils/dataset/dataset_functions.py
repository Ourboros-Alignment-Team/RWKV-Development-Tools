import torch
from .dataset import MultimodalDataset
from concurrent.futures import ThreadPoolExecutor
import random
from typing import List, Tuple, Any

class MyDataloader:
    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size,
        num_workers,
        multi_scale_alpha=1,
        infinite_loop_mode=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.n_dataset_ctx = dataset.total_ctx
        self.ctx_len = dataset.ctx_len
        self.infinite_loop_mode = infinite_loop_mode

        assert (
            0.1 < multi_scale_alpha <= 1
        ), f"multi_scale_alpha must be in (0.1,1], but got {multi_scale_alpha}"

        self.multi_scale_alpha = multi_scale_alpha

    def __iter__(self):
        """返回迭代器自身"""
        self.acc_ctx = 0
        return self

    def __next__(self):
        """并行返回下一个批次的数据"""
        if self.acc_ctx >= self.n_dataset_ctx:
            if self.infinite_loop_mode:
                self.acc_ctx = 0
            else:
                raise StopIteration

        assert self.ctx_len * self.multi_scale_alpha > 2

        ctx_len = random.randint(
            int(self.ctx_len * self.multi_scale_alpha),
            self.ctx_len,
        )

        futures = [
            (self.executor.submit(self.dataset.__getitem__, None, ctx_len))
            for _ in range(self.batch_size)
        ]
        # 收集所有结果
        batch_units = [future.result()[0] for future in futures if future is not None]
        batch_masks = [future.result()[1] for future in futures if future is not None]

        # 更新acc_idx
        self.acc_ctx += self.batch_size * ctx_len

        return batch_units, batch_masks

    def __del__(self):
        """确保线程池被正确关闭"""
        self.executor.shutdown()


class TraversalDataloader:
    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size,
        num_workers,
        multi_scale_alpha=1,
        infinite_loop_mode=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.n_dataset_ctx = dataset.total_ctx
        self.ctx_len = dataset.ctx_len
        self.infinite_loop_mode = infinite_loop_mode

        assert (
            0.1 < multi_scale_alpha <= 1
        ), f"multi_scale_alpha must be in (0.1,1], but got {multi_scale_alpha}"

        self.multi_scale_alpha = multi_scale_alpha

    def __iter__(self):
        """返回迭代器自身"""
        self.current_ctx = 0
        return self

    def __next__(self):

        assert self.ctx_len * self.multi_scale_alpha > 2

        ctx_len = random.randint(
            int(self.ctx_len * self.multi_scale_alpha),
            self.ctx_len,
        )

        if self.current_ctx >= self.n_dataset_ctx - ctx_len * self.batch_size:
            last_len = self.current_ctx - self.n_dataset_ctx - ctx_len * self.batch_size
            if last_len > self.ctx_len * self.batch_size // 2:
                ctx_len = last_len // self.batch_size
            elif self.infinite_loop_mode:
                self.current_ctx = 0
            else:
                raise StopIteration

        futures = [
            (
                self.executor.submit(
                    self.dataset.__getitem__,
                    self.current_ctx + i * (ctx_len - 1),
                    ctx_len,
                )
                if self.current_ctx + i * ctx_len < self.n_dataset_ctx
                else None
            )
            for i in range(self.batch_size)
        ]

        batch_units = [future.result()[0] for future in futures if future is not None]
        batch_masks = [future.result()[1] for future in futures if future is not None]

        self.current_ctx += self.batch_size * (ctx_len - 1)
        return batch_units, batch_masks

    def __del__(self):
        """确保线程池被正确关闭"""
        self.executor.shutdown()


class UnitStreamProcessor:
    def __init__(self, config):
        # self.voice_unit_len = (
        #     config.vocoder.head.hop_length * config.vocoder.adapter.chunk_len
        # )
        # assert config.vocoder.sample_rate % self.voice_unit_len == 0
        # self.voice_unit_rate = config.vocoder.sample_rate // self.voice_unit_len
        pass

    def encode(
        self, rwkv, feat_extractor, batch_units, vocode_and_adapt_func, device, dtype
    ):
        unit_dicts = {
            "main": torch.empty(
                (len(batch_units), 0, rwkv.args.n_embd),
                device=device,
                dtype=dtype,
            ),
            "text1_voice0_masks": [],
            "tokens_target": [],
            "voice_groups_target": [],
        }
        for batch_idx, units in enumerate(batch_units, 0):
            all_text_tokens = []
            accumulate_tokens = []
            text1_voice0_masks = []
            voice_groups_target = []
            accumulate_voice = torch.empty(
                (1, 2, 0), device=device, dtype=dtype
            )  # batch, channel, L
            main_track = torch.empty(
                1, 0, rwkv.args.n_embd, device=device, dtype=dtype
            )  # batch, N, ch
            for i, unit in enumerate(units, 0):
                if isinstance(unit, int) or (
                    isinstance(unit, torch.Tensor) and unit.size() == torch.Size([])
                ):
                    # print(type(unit))
                    # 单轨数据的文本token模态
                    accumulate_tokens.append(unit)  # 累积文本token
                    all_text_tokens.append(unit)
                    text1_voice0_masks.append(1)
                    # 如果下一个元素不是int或者这是最后一个元素
                    if i == len(units) - 1 or not (
                        isinstance(units[i + 1], int)
                        or (
                            isinstance(units[i + 1], torch.Tensor)
                            and units[i + 1].size() == torch.Size([])
                        )
                    ):
                        # 将累计的tokens通过token_embd_func转换为embedding
                        tokens_tensor = torch.tensor(
                            accumulate_tokens, device=device, dtype=torch.long
                        )  # 转换成tensor
                        token_embeds = rwkv.embedding(
                            tokens_tensor.unsqueeze(0)
                        )  # 获取token的embedding
                        main_track = torch.cat(
                            (main_track, token_embeds), dim=1
                        )  # 拼接到main_track
                        accumulate_tokens = []  # 清空token的累积
                elif isinstance(unit, torch.Tensor):
                    unit = unit.to(device=device)
                    text1_voice0_masks.append(0)
                    all_text_tokens.append(0)
                    # 单轨数据的音频模态
                    accumulate_voice = torch.cat(
                        (accumulate_voice, unit), dim=-1
                    )  # 按照L拼接音频数据
                    # 如果下一个元素不是tensor或者这是最后一个元素
                    if i == len(units) - 1 or not isinstance(
                        units[i + 1], torch.Tensor
                    ):
                        # 将累计的语音tensor通过vocoder转换为embedding
                        voice_embeds = vocode_and_adapt_func(
                            rwkv, feat_extractor, accumulate_voice
                        )  # 通过vocoder获取音频的embedding
                        voice_groups_target.append(
                            (accumulate_voice, i - voice_embeds.size(1) + 1, i + 1)
                        )  # 记录音频起止，之后加验证
                        main_track = torch.cat(
                            (main_track, voice_embeds), dim=1
                        )  # 拼接到main_track
                        # all_voice_groups.append(())
                        accumulate_voice = torch.empty(
                            1, 2, 0, device=device, dtype=dtype
                        )  # 清空语音的累积
                elif isinstance(unit, dict):
                    # 多轨数据，包括视频模态等
                    # 暂时不实现
                    pass
                elif isinstance(unit, tuple):
                    # 多轨数据，包括视频模态等
                    # 暂时不实现
                    pass

            # 将每个batch的结果存入unit_dicts
            unit_dicts["text1_voice0_masks"].append(text1_voice0_masks)
            unit_dicts["tokens_target"].append(all_text_tokens[1:])
            unit_dicts["voice_groups_target"].append(voice_groups_target)
            if batch_idx == 0:
                unit_dicts["main"] = main_track
            else:
                unit_dicts["main"] = torch.cat(
                    (unit_dicts["main"], main_track), dim=0
                )  # 拼接多个batch的结果
        return unit_dicts


class EpochSampleDataloader:
    def __init__(
        self,
        dataset_folder_list: List[str],
        n_sample_list: List[float],
        batch_size,
        num_workers,
        tokenizer=None,
        voice_read_func=None,
        video_load_func=None,
        ctx_len=3072,
        total_epoch=1,
        use_qa_mask=False,
    ):
        assert len(dataset_folder_list) == len(
            n_sample_list
        ), "dataset_folder_list and n_sample_list must have the same length"

        self.dataset_list = [
            MultimodalDataset(
                dataset_folder,
                tokenizer,
                voice_read_func,
                video_load_func,
                ctx_len=ctx_len,
                qa_mask_on=use_qa_mask,
            )
            for dataset_folder in dataset_folder_list
        ]
        self.n_sample_list = n_sample_list

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataloader_list = [
            (
                MyDataloader(ds, batch_size, num_workers, infinite_loop_mode=True)
                if samp > 0
                else TraversalDataloader(ds, batch_size, num_workers)
            )
            for ds, samp in zip(self.dataset_list, n_sample_list)
        ]
        self.total_epoch = total_epoch

    def __iter__(self):
        self.now_epoch = 0
        return self

    def __next__(self):
        self.now_epoch += 1
        if self.now_epoch > self.total_epoch:
            raise StopIteration
        epoch_units_list = []
        epoch_masks_list = []
        batch_list = []
        for dataloader, n_sample in zip(self.dataloader_list, self.n_sample_list):
            if n_sample > 0:
                for i, (batch_unit, batch_mask) in enumerate(dataloader):
                    for unit, mask in zip(batch_unit, batch_mask):
                        batch_list.append((unit, mask))
                    if i >= n_sample:
                        break
            else:
                for batch_unit, batch_mask in dataloader:
                    for unit, mask in zip(batch_unit, batch_mask):
                        batch_list.append((unit, mask))
        random.shuffle(batch_list)

        while len(batch_list) > 0:
            units, masks = zip(*batch_list[: self.batch_size])
            epoch_units_list.append(list(units))
            epoch_masks_list.append(list(masks))
            batch_list = batch_list[self.batch_size :]
        return epoch_units_list, epoch_masks_list

    def __del__(self):
        for dataloader in self.dataloader_list:
            del dataloader


def rl_collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[List[Any], ...]:
    """
    Custom collate function to batch the data returned by GSM8KRLDataset.

    Args:
        batch (List[Tuple[Any, ...]]): A list of tuples, where each tuple contains the data returned by __getitem__.

    Returns:
        Tuple[List[Any], ...]: A tuple of 6 lists, each containing batched data for one of the variables.
    """
    # Unpack the batch into separate lists
    input_conversations_batch = [item[0] for item in batch]
    resp_start_with_tokens_batch = [item[1] for item in batch]
    cleaned_answer_batch = [item[2] for item in batch]
    ground_truth_batch = [item[3] for item in batch]
    begin_with_state_batch = [item[4] for item in batch]
    kwargs_batch = {}
    for b in batch:
        for k, v in b[5].items():
            if f"{k}_batch" not in kwargs_batch:
                kwargs_batch[f"{k}_batch"] = []
            kwargs_batch[f"{k}_batch"].append(v)
            print(k, v)
    
    # [item[5] for item in batch]

    return (
        input_conversations_batch,
        resp_start_with_tokens_batch,
        ground_truth_batch,
        cleaned_answer_batch,
        begin_with_state_batch,
        kwargs_batch,
    )
    