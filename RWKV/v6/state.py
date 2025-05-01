import torch
import copy


class LayerRWKVStates:
    def __init__(
        self,
        tmix_shift_states,
        tmix_wkv_states,
        cmix_shift_states,
    ):
        self.tmix_shift_states = tmix_shift_states
        self.tmix_wkv_states = tmix_wkv_states
        self.cmix_shift_states = cmix_shift_states


class RWKVStates:
    def __init__(self, tmix_shift_states, tmix_wkv_states, cmix_shift_states):
        self.tmix_shift_states = tmix_shift_states
        self.tmix_wkv_states = tmix_wkv_states
        self.cmix_shift_states = cmix_shift_states

    def clone(self):
        return RWKVStates(
            self.tmix_shift_states.detach().clone(),
            self.tmix_wkv_states.detach().clone(),
            self.cmix_shift_states.detach().clone(),
        )

    @torch.no_grad()
    def duplicate(self, times):
        tmix_shift_states_list = []
        tmix_wkv_states_list = []
        cmix_shift_states_list = []
        for _ in range(times):
            tmix_shift_states_list.append(self.tmix_shift_states)
            tmix_wkv_states_list.append(self.tmix_wkv_states)
            cmix_shift_states_list.append(self.cmix_shift_states)
        return RWKVStates(
            tmix_shift_states=torch.cat(tmix_shift_states_list, dim=2),
            tmix_wkv_states=torch.cat(tmix_wkv_states_list, dim=1),
            cmix_shift_states=torch.cat(cmix_shift_states_list, dim=2),
        )

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        tmix_shift_states = torch.empty((N, B, C), device=device, dtype=dtype)
        tmix_wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size), device=device, dtype=torch.float
        )
        cmix_shift_states = torch.empty((N, B, C), device=device, dtype=dtype)
        return RWKVStates(tmix_shift_states, tmix_wkv_states, cmix_shift_states)

    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = RWKVStates.empty(N, B, C, n_head, head_size, device, dtype)
        result.tmix_shift_states[:] = 0
        result.tmix_wkv_states[:] = 0
        result.cmix_shift_states[:] = 0
        return result

    @staticmethod
    def create_like(other, batch_size=None):
        N, B, n_head, head_size, head_size = other.tmix_wkv_states.size()
        _, _, C = other.tmix_shift_states.size()
        if batch_size is not None:
            B = batch_size
        return RWKVStates.create(
            N,
            B,
            C,
            n_head,
            head_size,
            other.tmix_shift_states.device,
            other.tmix_shift_states.dtype,
        )

    def __getitem__(self, layer: int | slice):
        if isinstance(layer, int):
            return LayerRWKVStates(
                self.tmix_shift_states[layer],
                self.tmix_wkv_states[layer],
                self.cmix_shift_states[layer],
            )
        elif isinstance(layer, slice):  # 切片功能
            new_tmix_shift_states = self.tmix_shift_states[layer]
            new_tmix_wkv_states = self.tmix_wkv_states[layer]
            new_cmix_shift_states = self.cmix_shift_states[layer]
            return RWKVStates(
                new_tmix_shift_states, new_tmix_wkv_states, new_cmix_shift_states
            )
        else:
            raise TypeError(f"Invalid index type: {type(layer)}")

    def __len__(self):
        return self.tmix_wkv_states.size(0) if self.tmix_wkv_states is not None else 0

    def get_layer_size(self):
        return self.__len__()

    def get_batch_size(self):
        return self.tmix_wkv_states.size(1) if self.tmix_wkv_states is not None else 0

    def batchof(self, batch_idx: int | slice):
        if isinstance(batch_idx, slice):
            return RWKVStates(
                self.tmix_shift_states[:, batch_idx, :],
                self.tmix_wkv_states[:, batch_idx, :, :, :],
                self.cmix_shift_states[:, batch_idx, :],
            )
        elif isinstance(batch_idx, int):
            assert (
                0 <= batch_idx < self.tmix_wkv_states.size(1)
            ), "dim_index 超出范围, 应在 [0, batch_size: {self.tmix_wkv_states.size(1) - 1}] 之间。"
            return RWKVStates(
                self.tmix_shift_states[:, batch_idx : batch_idx + 1, :],
                self.tmix_wkv_states[:, batch_idx : batch_idx + 1, :, :, :],
                self.cmix_shift_states[:, batch_idx : batch_idx + 1, :],
            )

    def __add__(self, other):
        if other is None:
            N, B, n_head, head_size, head_size = self.tmix_wkv_states.size()
            _, _, C = self.tmix_shift_states.size()
            return self.__add__(
                RWKVStates.create(
                    N,
                    B,
                    C,
                    n_head,
                    head_size,
                    self.tmix_shift_states.device,
                    self.tmix_shift_states.dtype,
                )
            )
        assert isinstance(other, RWKVStates)
        tmix_shift_states = torch.cat(
            [self.tmix_shift_states, other.tmix_shift_states], dim=1
        )
        tmix_wkv_states = torch.cat([self.tmix_wkv_states, other.tmix_wkv_states], dim=1)
        cmix_shift_states = torch.cat(
            [self.cmix_shift_states, other.cmix_shift_states], dim=1
        )
        return RWKVStates(tmix_shift_states, tmix_wkv_states, cmix_shift_states)

    def pop(self, batch_idx: int | slice=0):
        if isinstance(batch_idx, slice):
            new_tmix_shift_states = torch.cat(
                (
                    self.tmix_shift_states[:, :batch_idx.start, :],
                    self.tmix_shift_states[:, batch_idx.stop :, :],
                ),
                dim=1,
            )
            new_tmix_wkv_states = torch.cat(
                (
                    self.tmix_wkv_states[:, :batch_idx.start, :, :, :],
                    self.tmix_wkv_states[:, batch_idx.stop :, :, :, :],
                ),
                dim=1,
            )
            new_cmix_shift_states = torch.cat(
                (
                    self.cmix_shift_states[:, :batch_idx.start, :],
                    self.cmix_shift_states[:, batch_idx.stop :, :],
                ),
                dim=1,
            )
            get_tmix_shift_states = self.tmix_shift_states[
                :, batch_idx.start : batch_idx.stop, :
            ]
            get_tmix_wkv_states = self.tmix_wkv_states[
                :, batch_idx.start : batch_idx.stop, :, :, :
            ]
            get_cmix_shift_states = self.cmix_shift_states[
                :, batch_idx.start : batch_idx.stop, :
            ]
            self.tmix_shift_states = new_tmix_shift_states
            self.tmix_wkv_states = new_tmix_wkv_states
            self.cmix_shift_states = new_cmix_shift_states
        elif isinstance(batch_idx, int):
            if not (0 <= batch_idx < self.tmix_wkv_states.size(1)):
                raise ValueError(
                    f"dim_index 超出范围，应在 [0, batch_size: {self.tmix_wkv_states.size(1) - 1}] 之间。"
                )
            new_tmix_shift_states = torch.cat(
                (
                    self.tmix_shift_states[:, :batch_idx, :],
                    self.tmix_shift_states[:, batch_idx + 1 :, :],
                ),
                dim=1,
            )
            new_tmix_wkv_states = torch.cat(
                (
                    self.tmix_wkv_states[:, :batch_idx, :, :, :],
                    self.tmix_wkv_states[:, batch_idx + 1 :, :, :, :],
                ),
                dim=1,
            )
            new_cmix_shift_states = torch.cat(
                (
                    self.cmix_shift_states[:, :batch_idx, :],
                    self.cmix_shift_states[:, batch_idx + 1 :, :],
                ),
                dim=1,
            )
            get_tmix_shift_states = self.tmix_shift_states[:, batch_idx : batch_idx + 1, :]
            get_tmix_wkv_states = self.tmix_wkv_states[:, batch_idx : batch_idx + 1, :, :, :]
            get_cmix_shift_states = self.cmix_shift_states[:, batch_idx : batch_idx + 1, :]
            self.tmix_shift_states = new_tmix_shift_states
            self.tmix_wkv_states = new_tmix_wkv_states
            self.cmix_shift_states = new_cmix_shift_states
        return RWKVStates(
            get_tmix_shift_states,
            get_tmix_wkv_states,
            get_cmix_shift_states,
        )

    def unbind(self):
        _tmix_wkv = [
            x.unsqueeze(1) for x in torch.unbind(self.tmix_wkv_states, dim=1)
        ]
        _tmix_shift = [
            x.unsqueeze(1) for x in torch.unbind(self.tmix_shift_states, dim=1)
        ]
        _cmix_shift = [
            x.unsqueeze(1) for x in torch.unbind(self.cmix_shift_states, dim=1)
        ]
        res = [
            RWKVStates(
                tmix_shift_states,
                tmix_wkv_states,
                cmix_shift_states,
            )
            for tmix_shift_states, tmix_wkv_states, cmix_shift_states in zip(
                _tmix_shift, _tmix_wkv, _cmix_shift
            )
        ]
        return res

    def __setitem__(self, layer: int, state: LayerRWKVStates):
        self.tmix_shift_states[layer] = state.tmix_shift_states
        self.tmix_wkv_states[layer] = state.tmix_wkv_states
        self.cmix_shift_states[layer] = state.cmix_shift_states

    def cpu(self):
        tmix_shift_states = self.tmix_shift_states.detach().cpu()
        tmix_wkv_states = self.tmix_wkv_states.detach().cpu()
        cmix_shift_states = self.cmix_shift_states.detach().cpu()
        return RWKVStates(tmix_shift_states, tmix_wkv_states, cmix_shift_states)

    def cuda(self,device_id=0):
        tmix_shift_states = self.tmix_shift_states.to(f"cuda:{device_id}")
        tmix_wkv_states = self.tmix_wkv_states.to(f"cuda:{device_id}")
        cmix_shift_states = self.cmix_shift_states.to(f"cuda:{device_id}")
        return RWKVStates(tmix_shift_states, tmix_wkv_states, cmix_shift_states)
        

    def to(self, device):
        tmix_shift_states = self.tmix_shift_states.to(device)
        tmix_wkv_states = self.tmix_wkv_states.to(device)
        cmix_shift_states = self.cmix_shift_states.to(device)
        return RWKVStates(tmix_shift_states, tmix_wkv_states, cmix_shift_states)

    def to_rwkv_list_states(self):
        n_layer = len(self)
        state = [None] * n_layer * 3
        for i in range(n_layer):
            state[i * 3 + 0] = copy.deepcopy(self.tmix_shift_states[i])
            state[i * 3 + 1] = copy.deepcopy(self.tmix_wkv_states[i])
            state[i * 3 + 2] = copy.deepcopy(self.cmix_shift_states[i])
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
            layerstate = LayerRWKVStates(
                tmix_shift_states=copy.deepcopy(state[i * 3 + 0]),
                tmix_wkv_states=copy.deepcopy(state[i * 3 + 1]),
                cmix_shift_states=copy.deepcopy(state[i * 3 + 2]),
            )
            new_states[i] = layerstate
        return new_states
