# Copyright (c) Tile-AI.
# **During the period from December 1, 2024, to Mar 14, 2025, this project is
# subject to additional collaboration terms with Microsoft Corporation.**
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from tilelang v0.1.12 examples/deepseek_v32.

import tilelang
from tilelang import language as T

LOG2E = 1.44269504

# Narrower head blocks leave the GEMM warp tiling under its 16-row minimum.
HEADS_PER_BLOCK = 64


def padded_heads(heads: int) -> int:
    return tilelang.cdiv(heads, HEADS_PER_BLOCK) * HEADS_PER_BLOCK


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_mla_fwd(
    Q,
    KV,
    Indices,
    Sink,
    heads,
    dim,
    tail_dim,
    topk,
    sm_scale,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert topk % block_I == 0
    sm_scale = sm_scale * LOG2E

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.bfloat16
    accum_dtype = T.float32
    H_per_block = HEADS_PER_BLOCK
    replicate = padded_heads(heads) // H_per_block
    BI = block_I
    NI = topk // block_I
    D = dim
    D_tail = tail_dim

    Q: T.Tensor([batch, seq_len, heads, D + D_tail], dtype)  # type: ignore
    KV: T.Tensor([batch, seq_len_kv, 1, D + D_tail], dtype)  # type: ignore
    Indices: T.Tensor([batch, seq_len, 1, topk], T.int32)  # type: ignore
    Sink: T.Tensor([padded_heads(heads)], accum_dtype)  # type: ignore
    Output = T.empty([batch, seq_len, heads, D], dtype)
    Lse = T.empty([batch, seq_len, heads], accum_dtype)

    with T.Kernel(seq_len * replicate, batch, threads=threads) as (bx, by):
        Q_shared = T.alloc_shared([H_per_block, D], dtype)
        Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
        KV_shared = T.alloc_shared([BI, D], dtype)
        K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
        mask = T.alloc_fragment([BI], "bool")

        acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
        acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
        S_shared = T.alloc_shared([H_per_block, BI], dtype)
        sumexp = T.alloc_fragment([H_per_block], accum_dtype)
        sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
        alpha = T.alloc_fragment([H_per_block], accum_dtype)
        m_i = T.alloc_fragment([H_per_block], accum_dtype)
        m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
        top = T.alloc_fragment([H_per_block], accum_dtype)
        denom = T.alloc_fragment([H_per_block], accum_dtype)

        T.fill(acc_o, 0)
        T.fill(sumexp, 0)
        # Finite so a block with no valid key does not compute -inf - -inf.
        T.fill(m_i, -(2**30))

        b_i = by
        s_i = bx // replicate
        H0 = (bx % replicate) * H_per_block
        H1 = H0 + H_per_block

        T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared, disable_tma=True)
        T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared, disable_tma=True)

        for i_i in T.Pipelined(NI, num_stages=num_stages):
            for bi_i in T.Parallel(BI):
                mask[bi_i] = Indices[b_i, s_i, 0, i_i * BI + bi_i] >= 0

            for bi_i, d_i in T.Parallel(BI, D):
                KV_shared[bi_i, d_i] = KV[
                    b_i, T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0), 0, d_i
                ]
            for bi_i, d_i in T.Parallel(BI, D_tail):
                K_tail_shared[bi_i, d_i] = KV[
                    b_i, T.max(Indices[b_i, s_i, 0, i_i * BI + bi_i], 0), 0, D + d_i
                ]

            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(accum_dtype))
            T.gemm(
                Q_shared,
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.gemm(
                Q_tail_shared,
                K_tail_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(m_i, m_i_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
            T.reduce_sum(acc_s, sumexp_i, dim=1)
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

            T.copy(acc_s, S_shared)
            T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        # The sink joins the softmax denominator only; Lse is the log2 normalizer with it.
        for h_i in T.Parallel(H_per_block):
            top[h_i] = T.max(m_i[h_i] * sm_scale, Sink[H0 + h_i])
            denom[h_i] = sumexp[h_i] * T.exp2(m_i[h_i] * sm_scale - top[h_i]) + T.exp2(
                Sink[H0 + h_i] - top[h_i]
            )
        for h_i, d_i in T.Parallel(H_per_block, D):
            acc_o[h_i, d_i] = T.if_then_else(
                denom[h_i] > 0,
                acc_o[h_i, d_i] * T.exp2(m_i[h_i] * sm_scale - top[h_i]) / denom[h_i],
                0,
            )
        for h_i in T.Parallel(H_per_block):
            sumexp[h_i] = T.log2(denom[h_i]) + top[h_i]

        T.copy(acc_o, Output[b_i, s_i, H0:H1, :], disable_tma=True)
        T.copy(sumexp, Lse[b_i, s_i, H0:H1], disable_tma=True)

    return Output, Lse


@tilelang.jit
def sparse_mla_bwd_delta(Out, dO, block_ND=32, num_stages=5):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    heads, dim = T.const("heads, dim")

    dtype = T.bfloat16
    accum_dtype = T.float32

    Out: T.Tensor([batch, seq_len, heads, dim], dtype)  # type: ignore
    dO: T.Tensor([batch, seq_len, heads, dim], dtype)  # type: ignore
    Delta = T.empty([batch, seq_len, heads], accum_dtype)

    with T.Kernel(heads, T.ceildiv(seq_len, block_ND), batch) as (bx, by, bz):
        o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        delta = T.alloc_fragment([block_ND], accum_dtype)
        acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(dim, block_ND), num_stages=num_stages):
            T.copy(
                Out[
                    bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND
                ],
                o,
            )
            T.copy(
                dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND],
                do,
            )
            for i, j in T.Parallel(block_ND, block_ND):
                acc[i, j] += o[i, j] * do[i, j]
        T.reduce_sum(acc, delta, 1)
        T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return Delta


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def sparse_mla_bwd(
    Q,
    KV,
    dO,
    Indices,
    Lse,
    Delta,
    dKV,
    heads,
    dim,
    tail_dim,
    topk,
    sm_scale,
    block_size=32,
    num_stages=0,
    threads=256,
):
    assert topk % block_size == 0
    sm_scale_log2 = sm_scale * LOG2E

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.bfloat16
    accum_dtype = T.float32
    block_H = HEADS_PER_BLOCK
    NH = tilelang.cdiv(heads, block_H)
    BS = block_size
    NS = topk // block_size
    D = dim
    D_tail = tail_dim
    split_store = 2

    Q: T.Tensor([batch, seq_len, heads, D + D_tail], dtype)  # type: ignore
    KV: T.Tensor([batch, seq_len_kv, 1, D + D_tail], dtype)  # type: ignore
    dO: T.Tensor([batch, seq_len, heads, D], dtype)  # type: ignore
    Indices: T.Tensor([batch, seq_len, 1, topk], T.int32)  # type: ignore
    Lse: T.Tensor([batch, seq_len, NH * block_H], accum_dtype)  # type: ignore
    Delta: T.Tensor([batch, seq_len, NH * block_H], accum_dtype)  # type: ignore
    dKV: T.Tensor([batch, seq_len_kv, 1, D + D_tail], accum_dtype)  # type: ignore
    dQ = T.empty([batch, seq_len, heads, D + D_tail], dtype)

    with T.Kernel(seq_len, batch, NH, threads=threads) as (s_i, by, bz):
        Q_shared = T.alloc_shared([block_H, D], dtype)
        Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
        KV_shared = T.alloc_shared([BS, D], dtype)
        KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
        dO_shared = T.alloc_shared([block_H, D], dtype)
        mask = T.alloc_fragment([BS], "bool")

        P_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dQ_shared = T.alloc_shared([block_H, D], dtype)
        dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

        acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
        acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
        acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
        acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
        acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
        acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

        T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
        T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
        T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

        T.clear(acc_dq)
        T.clear(acc_dq_tail)

        for i_i in T.Pipelined(NS, num_stages=num_stages):
            for bi_i in T.Parallel(BS):
                mask[bi_i] = Indices[by, s_i, 0, i_i * BS + bi_i] >= 0

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

            for bi_i, d_i in T.Parallel(BS, D):
                KV_shared[bi_i, d_i] = KV[
                    by, T.max(Indices[by, s_i, 0, i_i * BS + bi_i], 0), 0, d_i
                ]

            T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

            for bi_i, d_i in T.Parallel(BS, D_tail):
                KV_tail_shared[bi_i, d_i] = KV[
                    by, T.max(Indices[by, s_i, 0, i_i * BS + bi_i], 0), 0, D + d_i
                ]
            T.gemm(
                Q_tail_shared,
                KV_tail_shared,
                acc_p,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_p[h_i, bi_i] = T.if_then_else(
                    mask[bi_i],
                    T.exp2(acc_p[h_i, bi_i] * sm_scale_log2 - Lse[by, s_i, bz * block_H + h_i]),
                    0,
                )

            T.copy(acc_p, P_shared_cast)

            T.gemm(
                dO_shared,
                KV_shared,
                acc_dp,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_dp[h_i, bi_i] = (
                    acc_p[h_i, bi_i]
                    * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i])
                    * sm_scale
                )

            T.copy(acc_dp, dP_shared_cast)
            T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

            T.gemm(
                dP_shared_cast,
                Q_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.gemm(
                P_shared_cast,
                dO_shared,
                acc_dkv,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            T.clear(acc_dkv_tail)
            T.gemm(
                dP_shared_cast,
                Q_tail_shared,
                acc_dkv_tail,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            for s in range(split_store):
                for bi_i, d_i in T.Parallel(BS, D):
                    if bi_i < BS // split_store:
                        acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    if bi_i < BS // split_store:
                        acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                            bi_i + s * (BS // split_store), d_i
                        ]

                for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                    if Indices[by, s_i, 0, i_i * BS + bi_i + s * (BS // split_store)] >= 0:
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[by, s_i, 0, i_i * BS + bi_i + s * (BS // split_store)],
                                0,
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                    if Indices[by, s_i, 0, i_i * BS + bi_i + s * (BS // split_store)] >= 0:
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[by, s_i, 0, i_i * BS + bi_i + s * (BS // split_store)],
                                0,
                                D + d_i * 4,
                            ],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

        T.copy(acc_dq, dQ_shared)
        T.copy(acc_dq_tail, dQ_tail_shared)

        T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
        T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return dQ


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def lightning_indexer_logits(
    IndexQ,
    IndexK,
    Weights,
    KeyStarts,
    KeyEnds,
    block_N=128,
    num_stages=2,
    threads=512,
):
    heads, index_dim = T.const("heads, index_dim")
    block_Q = 128 // heads

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    IndexQ: T.Tensor([seq_len * heads, index_dim], dtype)  # type: ignore
    IndexK: T.Tensor([seq_len_kv, index_dim], dtype)  # type: ignore
    Weights: T.Tensor([seq_len, heads], accum_dtype)  # type: ignore
    KeyStarts: T.Tensor([seq_len], index_dtype)  # type: ignore
    KeyEnds: T.Tensor([seq_len], index_dtype)  # type: ignore
    Logits = T.empty([seq_len, seq_len_kv], accum_dtype)

    with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
        index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
        index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
        s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
        s_reshaped = T.reshape(s, (block_N, block_Q, heads))
        logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
        weights = T.alloc_fragment([block_Q, heads], accum_dtype)

        seq_len_i = bx * block_Q

        k_start = T.alloc_var(index_dtype)
        k_end = T.alloc_var(index_dtype)
        k_start = 2147483647
        k_end = -2147483648
        for bq_i in T.serial(block_Q):
            k_start = T.min(k_start, T.min(KeyStarts[seq_len_i + bq_i], seq_len_kv))
        for bq_i in T.serial(block_Q):
            k_end = T.max(k_end, T.min(KeyEnds[seq_len_i + bq_i], seq_len_kv))

        T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
        T.copy(Weights[seq_len_i, 0], weights)

        for nbn_i in T.Pipelined(T.ceildiv(k_end - k_start, block_N), num_stages=num_stages):
            T.copy(IndexK[k_start + nbn_i * block_N, 0], index_k_shared)
            T.gemm(
                index_k_shared,
                index_q_shared,
                s,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                s_reshaped[bn_i, bq_i, h_i] = (
                    T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                )
            T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)
            for bq_i, bn_i in T.Parallel(block_Q, block_N):
                Logits[seq_len_i + bq_i, k_start + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return Logits
