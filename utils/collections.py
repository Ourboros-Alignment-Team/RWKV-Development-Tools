import sys
import requests
from io import StringIO
import re
import torch, math


def parse_and_assign(input_string: str):
    variables = {}
    # 根据空格分割字符串，并去除首尾空格
    parts = input_string.strip().split()
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            # 获取变量名和值
            var_name = parts[i][2:]
            if i + 1 < len(parts):
                var_value = parts[i + 1]
                try:
                    var_value = int(var_value)
                except ValueError:
                    # 如果解析失败，则保留为字符串
                    pass
                # 存储变量名和值
                variables[var_name] = var_value
                # 删除已解析的部分
                del parts[i : i + 2]
            else:
                # 如果没有值，设为 None
                variables[var_name] = None
                # 删除已解析的部分
                del parts[i]
        else:
            i += 1
    return variables, " ".join(parts)


def parse_format_constrain_str(format_constrain_str: str):
    result = []
    pos = 0  # 当前处理字符串的位置

    # 正则表达式定义
    select_pattern = re.compile(
        r"(?<!\\)<([^<>]+)>"
    )  # 匹配 <选项1/选项2/...> 格式，不包括被转义的
    generate_pattern = re.compile(
        r"(?<!\\)\$(\d+)(?:->(\d+(?:,\d+)*))?(?:->(\d+(?:,\d+)*))?\$"
    )  # 匹配 $a->b,c,d,... 或 $a->b,c,d->x,y,z$
    escape_pattern = re.compile(r"\\([\\$<>])")  # 匹配转义符号 \$, \<, \\

    while pos < len(format_constrain_str):
        # 匹配 <选项1/选项2/……> 格式
        select_match = select_pattern.search(format_constrain_str, pos)
        generate_match = generate_pattern.search(format_constrain_str, pos)

        # 找到最近的匹配项
        if select_match and (
            not generate_match or select_match.start() < generate_match.start()
        ):
            if select_match.start() > pos:
                # 添加 select 之前的普通字符串
                substr = format_constrain_str[pos : select_match.start()]
                # 处理字符串中的转义字符
                result.append(
                    {
                        "type": "str",
                        "text": re.sub(escape_pattern, lambda m: m.group(1), substr),
                    }
                )

            # 解析选项
            options = select_match.group(1).split("/")
            if len(options) >= 2:
                # 处理每个选项中的转义字符
                options = [
                    re.sub(escape_pattern, lambda m: m.group(1), opt) for opt in options
                ]
                result.append({"type": "select", "selections": options})

            pos = select_match.end()

        # 匹配 $a->b,c,d,... 或 $a->b,c,d->x,y,z$ 格式
        elif generate_match:
            if generate_match.start() > pos:
                # 添加 generate 之前的普通字符串
                substr = format_constrain_str[pos : generate_match.start()]
                # 处理字符串中的转义字符
                result.append(
                    {
                        "type": "str",
                        "text": re.sub(escape_pattern, lambda m: m.group(1), substr),
                    }
                )

            # 解析 generate 语法
            n_max = int(generate_match.group(1))
            # 如果有第一个 "->" 右侧的内容，则解析 stop 列表，否则为空
            stop_values = (
                list(map(int, generate_match.group(2).split(",")))
                if generate_match.group(2)
                else []
            )
            # 如果有第二个 "->" 右侧的内容，则解析 supplement 列表，否则为空
            supplement_values = (
                list(map(int, generate_match.group(3).split(",")))
                if generate_match.group(3)
                else []
            )

            result.append(
                {
                    "type": "generate",
                    "n_max": n_max,
                    "stop": stop_values,
                    "supplement": supplement_values,
                }
            )

            pos = generate_match.end()

        # 如果都不匹配，则处理普通字符串
        else:
            substr = format_constrain_str[pos:]
            # 处理字符串中的转义字符
            result.append(
                {
                    "type": "str",
                    "text": re.sub(escape_pattern, lambda m: m.group(1), substr),
                }
            )
            break

    return result


class PrintInterceptor:
    def __init__(self, server: str):
        self.old_stdout = sys.stdout
        self.server = server

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

    def write(self, text):
        self.old_stdout.write(text)  # 在原始标准输出中打印
        if text.strip():  # 如果文本不是空白，则发送到服务器
            self.send_to_server({"message": text.strip()})

    def flush(self):
        self.old_stdout.flush()

    def send_to_server(self, data):
        try:
            requests.post(self.server, json=data)
        except Exception as e:
            self.old_stdout.write(f"\nError sending data: {e}")
            self.old_stdout.flush()


def extract_code(text: str):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return False


def pad_2d_list_with_zeros(data: list):
    # 获取每一行的长度，找到最长的一行
    row_lengths = [len(row) for row in data]
    max_length = max(row_lengths) if row_lengths else 0

    # 对每一行进行补零操作，使其长度与最长行相同
    padded_data = [row + [0] * (max_length - len(row)) for row in data]

    return padded_data, row_lengths


def pad_and_batch(data_list, batch_size, pad_value=0):
    # 获取列表的总长度
    total_len = len(data_list)

    # 计算每批次的长度
    batch_len = math.ceil(total_len / batch_size)

    # 计算需要填充的数量，确保总长度是 batch_size * batch_len 的整数倍
    pad_len = batch_size * batch_len - total_len

    # 填充 pad_value
    data_list += [pad_value] * pad_len

    # 将列表切片，划分为 batch_size 大小的二维列表
    batched_list = [
        data_list[i * batch_len : (i + 1) * batch_len] for i in range(batch_size)
    ]

    return batched_list


def pad_and_batch_tensor(tensor, batch_size):
    # 获取原始张量的形状
    original_shape = tensor.shape
    dim = original_shape[1]  # N是我们要批处理的那个维度

    # 计算新的N使其能被batch_size整除
    new_dim = (dim + batch_size - 1) // batch_size * batch_size
    pad_size = new_dim - dim  # 需要补0的数量

    # 如果有需要，在第二个维度上补0
    if pad_size > 0:
        pad_shape = (0, 0) * (tensor.ndim - 2) + (0, pad_size)  # 只在第二个维度补0
        tensor = torch.nn.functional.pad(tensor, pad_shape)

    # 重新计算并调整张量的形状
    reshaped_shape = (batch_size, new_dim // batch_size) + original_shape[2:]
    reshaped_tensor = tensor.view(reshaped_shape)

    return reshaped_tensor
