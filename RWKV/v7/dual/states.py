class LayerState:
    """管理单个层的状态类"""
    
    def __init__(self, prev_s=None, prev_sa=None):
        """
        初始化层状态
        
        Args:
            prev_s: 前一个s状态
            prev_sa: 前一个sa状态
        """
        self.prev_s = prev_s
        self.prev_sa = prev_sa
    
    def get(self):
        """
        获取状态元组
        
        Returns:
            tuple: (prev_s, prev_sa) 状态元组
        """
        return (self.prev_s, self.prev_sa)
    
    def update(self, prev_s, prev_sa):
        """
        更新状态
        
        Args:
            prev_s: 新的s状态
            prev_sa: 新的sa状态
            
        Returns:
            self: 支持链式调用
        """
        self.prev_s = prev_s
        self.prev_sa = prev_sa
        return self
    
    def to(self, device=None, dtype=None):
        """
        将状态转移到指定设备和数据类型
        
        Args:
            device: 目标设备
            dtype: 目标数据类型
            
        Returns:
            self: 支持链式调用
        """
        if self.prev_s is not None:
            self.prev_s = self.prev_s.to(device=device, dtype=dtype) if device is not None or dtype is not None else self.prev_s
        if self.prev_sa is not None:
            self.prev_sa = self.prev_sa.to(device=device, dtype=dtype) if device is not None or dtype is not None else self.prev_sa
        return self
    
    def cpu(self):
        """
        将状态转移到CPU
        
        Returns:
            self: 支持链式调用
        """
        return self.to(device="cpu")
    
    def clone(self):
        """
        克隆当前状态
        
        Returns:
            LayerState: 新的状态对象
        """
        if self.prev_s is not None and self.prev_sa is not None:
            return LayerState(self.prev_s.clone(), self.prev_sa.clone())
        return LayerState(self.prev_s, self.prev_sa)
    
    def detach(self):
        """
        分离状态的计算图
        
        Returns:
            self: 支持链式调用
        """
        if self.prev_s is not None:
            self.prev_s = self.prev_s.detach()
        if self.prev_sa is not None:
            self.prev_sa = self.prev_sa.detach()
        return self
    
    def __bool__(self):
        """
        检查状态是否有效
        
        Returns:
            bool: 如果prev_s和prev_sa都不为None，则返回True
        """
        return self.prev_s is not None and self.prev_sa is not None

class BlockStates:
    """
    管理RWKV模型中所有层的状态
    
    每一层的状态由LayerState对象管理，包含:
    - prev_s: 前一个状态矩阵
    - prev_sa: 前一个状态注意力向量
    """
    def __init__(self):
        # 使用字典存储各层状态，键为layer_id
        self.states = {}
    
    def __getitem__(self, layer_id):
        """
        获取指定层的状态
        
        Args:
            layer_id: 层ID
            
        Returns:
            LayerState: 层状态对象，如果不存在则返回None
        """
        return self.states.get(layer_id, None)
    
    def __setitem__(self, layer_id, value):
        """
        设置指定层的状态
        
        Args:
            layer_id: 层ID
            value: LayerState对象或(prev_s, prev_sa)元组
        """
        if isinstance(value, tuple) and len(value) == 2:
            # 如果是元组，转换为LayerState对象
            prev_s, prev_sa = value
            self.states[layer_id] = LayerState(prev_s, prev_sa)
        elif isinstance(value, LayerState):
            # 如果已经是LayerState对象，直接存储
            self.states[layer_id] = value
        else:
            raise TypeError("状态值必须是LayerState对象或(prev_s, prev_sa)形式的元组")
    
    def reset(self):
        """重置所有层的状态"""
        self.states = {}
    
    def to(self, device=None, dtype=None):
        """
        将所有状态转移到指定设备和数据类型
        
        Args:
            device: 目标设备
            dtype: 目标数据类型
            
        Returns:
            self: 支持链式调用
        """
        for layer_id, state in self.states.items():
            if state:  # 使用LayerState的__bool__方法检查状态是否有效
                state.to(device=device, dtype=dtype)
        return self
    
    def cpu(self):
        """
        将所有状态转移到CPU
        
        Returns:
            self: 支持链式调用
        """
        return self.to(device="cpu")
    
    def items(self):
        """
        返回所有层及其状态
        
        Returns:
            dict_items: (layer_id, LayerState) 键值对
        """
        return self.states.items()
    
    def get_tuple_states(self):
        """
        获取所有层的状态元组形式，用于兼容旧代码
        
        Returns:
            dict: {layer_id: (prev_s, prev_sa)} 形式的字典
        """
        return {layer_id: state.get() for layer_id, state in self.states.items()}
    
    def clone(self):
        """
        克隆当前所有状态
        
        Returns:
            BlockStates: 新的BlockStates对象，包含所有状态的克隆
        """
        new_states = BlockStates()
        for layer_id, state in self.states.items():
            new_states.states[layer_id] = state.clone()
        return new_states
    
    def detach(self):
        """
        分离所有状态的计算图
        
        Returns:
            self: 支持链式调用
        """
        for state in self.states.values():
            state.detach()
        return self
    
    def __len__(self):
        """返回已保存状态的层数"""
        return len(self.states)

