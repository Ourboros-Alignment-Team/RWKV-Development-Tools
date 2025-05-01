import torch
import copy


class LayerRWKVStates:
    def __init__(
        self,
        time_mix_state: tuple[torch.Tensor, torch.Tensor],
        channel_mix_state: torch.Tensor,
    ):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class RWKVStates:
    def __init__(self, shift_states, wkv_states):
        self.shift_states = shift_states
        self.wkv_states = wkv_states

    def clone(self):
        return RWKVStates(
            self.shift_states.detach().clone(), self.wkv_states.detach().clone()
        )

    @torch.no_grad()
    def duplicate(self, times):
        wkv_states_list = []
        shift_states_list = []
        for _ in range(times):
            wkv_states_list.append(copy.deepcopy(self.wkv_states))
            shift_states_list.append(copy.deepcopy(self.shift_states))
        return RWKVStates(
            shift_states=torch.cat(shift_states_list, dim=2),
            wkv_states=torch.cat(wkv_states_list, dim=1),
        )

    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = RWKVStates.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def create_like(other, batch_size=None):
        N, B, n_head, head_size, head_size = other.wkv_states.size()
        _, _, _, C = other.shift_states.size()
        if batch_size is not None:
            B = batch_size
        return RWKVStates.create(
            N,
            B,
            C,
            n_head,
            head_size,
            other.shift_states.device,
            other.shift_states.dtype,
        )

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size), device=device, dtype=torch.float
        )
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return RWKVStates(shift_states, wkv_states)

    @staticmethod
    def merge(s1, s2, s1_part: float = 0.5, s2_part: float = 0.5):
        wkv_states = s1.wkv_states * s1_part + s2.wkv_states * s2_part
        shift_states = s1.shift_states * s1.s1_part + s2.shift_states * s2_part
        return RWKVStates(shift_states, wkv_states)

    def decay(self, ratio: float = 0.95):
        if ratio == 0:
            return self
        self.wkv_states = self.wkv_states * ratio
        self.shift_states = self.shift_states
        return self

    def __getitem__(self, layer: int):
        if isinstance(layer, int):
            return LayerRWKVStates(
                (self.shift_states[layer, 0], self.wkv_states[layer]),
                (self.shift_states[layer, 1]),
            )
        elif isinstance(layer, slice):  # 切片功能
            new_wkv_states = self.wkv_states[layer]
            new_shift_states = self.shift_states[layer]
            return RWKVStates(new_shift_states, new_wkv_states)
        else:
            raise TypeError(f"Invalid index type: {type(layer)}")

    def __len__(self):
        return self.wkv_states.size(0) if self.wkv_states is not None else 0

    def get_batch_size(self):
        return self.wkv_states.size(1) if self.wkv_states is not None else 0

    def batchof(self, batch_idx):
        # if not (0 <= batch_idx < self.wkv_states.size(1)):
        #     raise ValueError(
        #         f"dim_index 超出范围，应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
        #     )
        if isinstance(batch_idx, slice):
            return RWKVStates(
                self.shift_states[:, :, batch_idx, :],
                self.wkv_states[:, batch_idx, :, :, :],
            )
        elif isinstance(batch_idx, int):
            assert (
                0 <= batch_idx < self.wkv_states.size(1)
            ), "dim_index 超出范围, 应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
            return RWKVStates(
                self.shift_states[:, :, batch_idx : batch_idx + 1, :],
                self.wkv_states[:, batch_idx : batch_idx + 1, :, :, :],
            )

    def __add__(self, other):
        if other is None:
            N, B, n_head, head_size, head_size = self.wkv_states.size()
            _, _, _, C = self.shift_states.size()
            return self.__add__(
                RWKVStates.create(
                    N,
                    B,
                    C,
                    n_head,
                    head_size,
                    self.shift_states.device,
                    self.shift_states.dtype,
                )
            )
        assert isinstance(other, RWKVStates)
        wkv_states = torch.cat([self.wkv_states, other.wkv_states], dim=1)
        shift_states = torch.cat([self.shift_states, other.shift_states], dim=2)
        return RWKVStates(shift_states, wkv_states)

    def remove_at(self, batch_idx):
        if not (0 <= batch_idx < self.wkv_states.size(1)):
            raise ValueError(
                f"dim_index 超出范围，应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
            )
        new_wkv_states = torch.cat(
            (
                self.wkv_states[:, :batch_idx, :, :, :],
                self.wkv_states[:, batch_idx + 1 :, :, :, :],
            ),
            dim=1,
        )
        new_shift_states = torch.cat(
            (
                self.shift_states[:, :, :batch_idx, :],
                self.shift_states[:, :, batch_idx + 1 :, :],
            ),
            dim=2,
        )
        return RWKVStates(new_shift_states, new_wkv_states)

    def unbind(self):
        _wkv = [x.unsqueeze(1) for x in torch.unbind(self.wkv_states, dim=1)]
        _shift = [x.unsqueeze(2) for x in torch.unbind(self.shift_states, dim=2)]
        res = [
            RWKVStates(shift_states, wkv_states)
            for wkv_states, shift_states in zip(_wkv, _shift)
        ]
        return res

    def __setitem__(self, layer: int, state: LayerRWKVStates):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

    def cpu(self):
        wkv_states = self.wkv_states.detach().cpu()
        shift_states = self.shift_states.detach().cpu()
        return RWKVStates(shift_states, wkv_states)

    def cuda(self):
        wkv_states = self.wkv_states.to("cuda")
        shift_states = self.shift_states.to("cuda")
        return RWKVStates(shift_states, wkv_states)

    def to(self, device):
        wkv_states = self.wkv_states.to(device)
        shift_states = self.shift_states.to(device)
        return RWKVStates(shift_states, wkv_states)

    @classmethod
    def save(cls, item, path):
        item = item.cpu()
        data = {"wkv_states": item.wkv_states, "shift_states": item.shift_states}
        torch.save(data, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu")
        wkv_states = data["wkv_states"]
        shift_states = data["shift_states"]
        item = cls(shift_states, wkv_states)
        return item

    def to_rwkv_list_states(self):
        n_layer = len(self)
        state = [None] * n_layer * 3
        for i in range(n_layer):
            state[i * 3 + 0] = copy.deepcopy(self.shift_states[i, 0])
            state[i * 3 + 1] = copy.deepcopy(self.wkv_states[i])
            state[i * 3 + 2] = copy.deepcopy(self.shift_states[i, 1])
        return state

    @staticmethod
    def from_rwkv_list_states(state):
        n_layer = len(state) // 3
        new_states = RWKVStates.empty(
            N=n_layer,
            B=state[0].size(0),
            C=state[0].size(1),
            n_head=state[1].size(1),
            head_size=state[1].size(2),
            device=state[0].device,
            dtype=state[0].dtype,
        )
        for i in range(n_layer):
            blockstate = LayerRWKVStates(
                (copy.deepcopy(state[i * 3 + 0]), copy.deepcopy(state[i * 3 + 1])),
                copy.deepcopy(state[i * 3 + 2]),
            )
            new_states[i] = blockstate
        return new_states
