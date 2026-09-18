"""Machine-readable HBServe capability boundary for workload selection."""

from __future__ import annotations

from typing import Any


CAPABILITY_SCHEMA = {"name": "hbserve.capabilities", "version": 4}


def current_capabilities() -> dict[str, Any]:
    """Return both execution modes; the remaining fields describe closed-loop serving."""

    return {
        "schema": CAPABILITY_SCHEMA,
        "execution_modes": {
            "closed_loop": {
                "input": "--requests",
                "request_scheduling_feedback": True,
                "compute_time": "timing_provider",
                "ttft_tpot": "only_with_compute",
            },
            "fixed_window": {
                "input": "--experiment",
                "request_scheduling_feedback": False,
                "compute_time": "not_modeled",
                "ttft_tpot": "not_reported",
                "same_logical_trace_across_topologies": True,
                "preflight_without_simulator": True,
                "static_direct_placement_policies": [
                    "capacity_balanced", "weights_first", "kv_first", "profiled_hotset",
                ],
                "independent_training_profile": "python -m hbserve profile",
                "direct_placement_migration": False,
                "peer_KV_policies": ["static", "capacity_migration"],
                "peer_KV_initial_state": "empty_born_on_first_write",
                "peer_KV_preflight": "requires_native_resolved_capacity",
                "tiered_fill_and_dirty_writeback": True,
                "hbm_fronted_backing": ["hbf", "external"],
                "cache_policies": ["address_only_lru", "decayed_lfu", "threshold_promotion", "class_aware"],
                "per_topology_mapping_configuration": True,
            },
        },
        "request_fields": [
            "request_id",
            "arrival_ns",
            "model_id",
            "prompt_tokens",
            "output_tokens",
            "token_ids_optional",
            "cache_salt",
        ],
        "conversation_ancestry": False,
        "prefix_block_hash_identity": True,
        "prefix_cache_lifecycle": True,
        "prefix_cache": {"tier": "hbm", "bounded_capacity": True, "ttl": True,
                         "model_scope": "dense_only_until_per_prefix_MoE_route_identity_is_supported",
                         "identity": "model_digest_parent_hash_tokens_cache_salt",
                         "publication": "full_prompt_blocks_after_physical_batch_completion",
                         "storage": "reference_counted_shared_KV_blocks", "persistent": False,
                         "requires_explicit_token_ids": True},
        "scheduler": "token_budgeted_mixed_iteration_continuous_batcher_v1",
        "kv_allocation_policy": "paged_blocks_incremental_v1",
        "kv_placement_targets": {"hot": ["hbm"], "cold": ["hbf", "external"]},
        "kv_migration_policy": "whole_request_coldest_first_v1",
        "preemption_policy": "youngest_request_first_swap_or_recompute_v1",
        "timing_models": ["gpu_calibrated", "roofline", "memory_only", "linear"],
        "gpu_calibration": {"runtime": "hbserve.gpu_profile", "operators": "packed_W8_paged_attention",
                            "overlap": "fixed_plus_max_compute_memory_per_operator",
                            "shape": "per_request_query_context_and_per_layer_expert_histogram"},
        "compute_prefetch_overlap": "known_next_layer_memory_during_compute",
        "selected_expert_availability": "current_layer_routing_ready",
        "physical_feedback": True,
        "dense_models": True,
        "moe_models": True,
        "model_catalog_converter": "hbserve.catalog",
    }
