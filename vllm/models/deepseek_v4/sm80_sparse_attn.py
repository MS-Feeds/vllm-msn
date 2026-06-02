        Each decode step calls this layer's gather many times with
        identical (length, device) tuples; the cache avoids the per-call
        allocation and kernel launch."""
        key = (n, device)
        cached = self._arange_cache.get(key)
        if cached is None:
            cached = torch.arange(0, n, device=device)
            self._arange_cache[key] = cached
        return cached

    def _ref_sparse_attn_decode_gather(
        self,
        q: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_block_size: int,
        swa_indices: torch.Tensor,
        swa_topk_length: torch.Tensor | None,
        attn_sink: torch.Tensor | None,
        extra_kv_cache: torch.Tensor | None,
        extra_block_size: int,
        extra_indices: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        """SM80 reference decode: gather-then-dequantise only the topk
        positions, then dispatch to the split-K attention kernel.

        Gather + invalid-mask construction is a single fused kernel per
        scope (`gather_dequant_two_scopes_with_mask`), writing into a
        merged ``(B, swa_topk + extra_topk, head_dim)`` output buffer so
        no torch.cat is needed when both SWA and compressed-KV scopes are
        present."""
        b, s_q, h_q, d_qk = q.shape
        d_v = self.head_dim
        bs = b * s_q

        # Flatten leading dims to (bs, topk_per_scope) so the kernel can
        # treat each token as one batch entry.
        swa_indices_2d = swa_indices.reshape(bs, -1)
        if extra_indices is not None:
            extra_indices_2d = extra_indices.reshape(bs, -1)
        else:
            extra_indices_2d = None

        gathered_kv_flat, invalid_flat = gather_dequant_two_scopes_with_mask(
            swa_kv_cache=swa_kv_cache,
            swa_block_size=swa_block_size,
            swa_indices=swa_indices_2d,
            swa_topk_length=swa_topk_length,
            extra_kv_cache=extra_kv_cache,
            extra_block_size=extra_block_size,
            extra_indices=extra_indices_2d,
            extra_topk_length=extra_topk_length,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            head_dim=d_qk,
        )

        if os.environ.get("DSV4_GATHER_VERIFY") == "1":
            # One-shot side-by-side: run the legacy chain and compare.
            # Set DSV4_GATHER_VERIFY=1 to enable; off by default. Logs to
            # /tmp/dsv4_gather_verify.log so it survives multiple TP workers
            # without interleaving stdout.
            self._verify_gather(
                q,
                swa_kv_cache,
                swa_block_size,
                swa_indices,
                swa_topk_length,
                extra_kv_cache,
                extra_block_size,
                extra_indices,
                extra_topk_length,
                gathered_kv_flat,
                invalid_flat,
                bs,
                d_qk,
            )

        # q may arrive non-contiguous from the upstream o_padded[...] slice.
        q_flat = q.view(bs, h_q, d_qk).to(torch.bfloat16).contiguous()

        out_flat = _dsv4_sm80_sparse_attn_decode_triton(
            q_flat,
            gathered_kv_flat,
            invalid_flat,
            attn_sink,
            self.scale,
            d_v,
        )
        # Match the prior PyTorch shape: (b, h_q, d_v) for s_q=1.
        return out_flat.view(b, h_q, d_v)

    def _verify_gather(
        self,
        q: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_block_size: int,
        swa_indices: torch.Tensor,
        swa_topk_length: torch.Tensor | None,
        extra_kv_cache: torch.Tensor | None,
        extra_block_size: int,
        extra_indices: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
        new_gathered: torch.Tensor,
        new_mask: torch.Tensor,
        bs: int,
        d_qk: int,
    ) -> None:
        """Run the legacy gather + invalid_mask + cat chain and compare
        against the fused output. Dumps diff stats and aborts on mismatch
        so the bug is caught early."""
        b, s_q = q.shape[0], q.shape[1]

        def _legacy_scope(kv_cache, block_size, indices, topk_length):
            indices_3d = indices.reshape(b, s_q, -1)
            topk = indices_3d.size(-1)
            gathered = self._gather_dequant_blocked_k_at_indices(
                kv_cache, indices_3d.reshape(-1), block_size
            ).view(b, s_q, topk, d_qk)
            mask = indices_3d == -1
            if topk_length is not None:
                topk_length = topk_length.reshape(b)
                ar = torch.arange(topk, device=mask.device)
                mask = mask | (ar.view(1, 1, topk) >= topk_length.view(b, 1, 1))
            return gathered, mask

        ref_g, ref_m = _legacy_scope(
            swa_kv_cache, swa_block_size, swa_indices, swa_topk_length
        )
        if extra_kv_cache is not None and extra_indices is not None:
            ex_g, ex_m = _legacy_scope(
                extra_kv_cache, extra_block_size, extra_indices, extra_topk_length
            )
            ref_g = torch.cat([ref_g, ex_g], dim=2)
            ref_m = torch.cat([ref_m, ex_m], dim=2)

        ref_g_flat = ref_g.view(bs, -1, d_qk)
        ref_m_flat = ref_m.view(bs, -1)

        rg = torch.nan_to_num(ref_g_flat.float(), 0.0, 0.0, 0.0)
        ng = torch.nan_to_num(new_gathered.float(), 0.0, 0.0, 0.0)
        nan_match = (torch.isnan(ref_g_flat) == torch.isnan(new_gathered)).all().item()
        g_diff = (rg - ng).abs().max().item()
        m_diff = (ref_m_flat != new_mask).any().item()
        m_diff_count = (ref_m_flat != new_mask).sum().item()

        with open("/tmp/dsv4_gather_verify.log", "a") as f:
            f.write(
                f"layer={self.prefix} bs={bs} swa={swa_indices.shape} "
                f"extra={None if extra_indices is None else extra_indices.shape} "
                f"g_max={g_diff:.3e} nan_match={nan_match} "
                f"m_diff={m_diff} m_count={m_diff_count}\n"
            )
        if (not nan_match) or g_diff > 1e-4 or m_diff:
            raise RuntimeError(
                f"DSV4 gather verify mismatch on layer {self.prefix}: "
                f"g_diff={g_diff} nan_match={nan_match} "
                f"m_diff={m_diff} m_count={m_diff_count}"
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            if use_dsv4_reference_kernels():
                # SM80/ROCm reference path. The reference returns the
                # attention output rather than writing to `out=`, so copy
                # into the output slice.
                output_chunk = self._ref_sparse_attn_prefill(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    topk_length=combined_lens,
                )
                output[query_start:query_end].copy_(output_chunk.to(output.dtype))
            else:
                output_chunk, _, _ = flash_mla_sparse_fwd(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output[query_start:query_end],
                )

    def _ref_sparse_attn_prefill(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pure-PyTorch sparse MLA prefill reference."""
        indices = indices.clone().squeeze(1)
        s_q, h_q, d_qk = q.shape
        topk = indices.shape[-1]
        s_kv = kv.shape[0]
        if topk_length is not None:
            mask = torch.arange(topk, device=indices.device).unsqueeze(
                0
            ) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0

        qf = q.float()
        gathered_kv = (
            kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
        )
        scores = qf @ gathered_kv.transpose(1, 2)
        scores *= self.scale
        scores[invalid_mask.unsqueeze(1).expand_as(scores)] = float("-inf")

        orig_lse = torch.logsumexp(scores, dim=-1)
        lse_for_o = orig_lse
        if self.attn_sink is not None:
            lse_for_o = torch.logsumexp(
                torch.stack(
                    [
                        orig_lse,
                        self.attn_sink[:h_q].view(1, h_q).expand_as(orig_lse),
                    ],
                    dim=0,
                ),
                dim=0,
            )
        lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float("+inf")
        probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
        out = probs @ gathered_kv[..., : self.head_dim]
        lonely_q_mask = orig_lse == float("-inf")
        out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
        return out.to(torch.bfloat16)


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
