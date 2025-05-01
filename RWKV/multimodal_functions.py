import torch

# def voice_encode_and_adapt(rwkv,feat_extractor, x):
#     l, r = torch.split(x, 1, dim=1)
#     vs = []
#     for v in (l, r):

#         B, ch, N = v.size()

#         # 假设采样率为24000Hz，计算音频时长
#         duration_in_seconds = N / 24000.0

#         # 如果音频时长超过1秒，计算切分段数
#         # TODO 随机切段不太行，需要按照24000切
#         if duration_in_seconds > 1:
#             v_segments = torch.split(v, 24000, dim=2)
#         else:
#             v_segments = [v]

#         processed_segments = []

#         # 处理每个切分的音频段
#         for segment in v_segments:
#             segment = feat_extractor(
#                 segment.to(
#                     device=next(feat_extractor.parameters()).device,
#                     dtype=next(feat_extractor.parameters()).dtype,
#                 )
#             ).to(
#                 device=next(rwkv.parameters()).device,
#                 dtype=next(rwkv.parameters()).dtype,
#             )
#             processed_segments.append(segment)

#         # 将处理后的段合并
#         v = torch.cat(processed_segments, dim=2)
#         vs.append(v)
#     x = rwkv.track_mixing(*vs)
#     x = rwkv.adapter_e(x)
#     return x