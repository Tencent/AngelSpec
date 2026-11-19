import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.offline import (
    HiddenDataset,
    HiddenWriter,
    OfflineBatchFetcher,
    OfflineRowPlan,
    prefetch_offline,
)
from angelspec.data.utils import DFlashPackingCollator, MTPPackingCollator
from angelspec.models.dflash import DFlashModel, _create_dflash_mask_mod
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.training.offline import (
    evaluate_offline,
    load_row_plan,
    log_offline_metrics,
    run_offline_training,
    validate_offline_args,
)
from angelspec.utils.metrics import token_weighted_loss_scale
from tests.offline_native import native_recipe, to_native


def feature(length, token):
    return {
        "input_ids": torch.full((1, length), token, dtype=torch.int64),
        "loss_mask": torch.ones(1, length, dtype=torch.uint8),
        "attention_mask": torch.ones(1, length, dtype=torch.bool),
        "hidden_states": torch.randn(1, length, 8),
        "last_hidden_states": torch.randn(1, length, 8),
    }


def make_config(**overrides):
    values = dict(
        hidden_size=8,
        target_hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        mask_token_id=15,
        num_target_layers=2,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        max_position_embeddings=512,
        attention_type="mla",
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        enable_hidden_correction=True,
    )
    values.update(overrides)
    return DFlashConfig(**values)


class TestOfflinePacking(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        recipe = native_recipe([1], 3, 8, 16)
        self.samples = [feature(n, i + 1) for i, n in enumerate([8, 4, 8, 12, 4])]
        writer = HiddenWriter(self.root, recipe)
        for i, tensors in enumerate(self.samples):
            writer.write(str(i), to_native(tensors, [1], 3))
        writer.publish([str(i) for i in range(len(self.samples))])
        self.dataset = HiddenDataset(self.root)

    def fetcher(self, rank=0, world=1, shuffle=False):
        return OfflineBatchFetcher(
            self.dataset,
            rank=rank,
            world_size=world,
            seed=42,
            max_length=12,
            hidden_size=8,
            aux_layer_ids=[1],
            final_layer_id=3,
            vocab_size=16,
            min_loss_tokens=4,
            shuffle=shuffle,
            packing=True,
        )

    def args(self, *overrides):
        config = load_config(
            cli_args=[
                f"offline.manifest_path={self.root}",
                "offline.frozen_target_path=/tmp/frozen",
                "offline.eval_samples=0",
                "model.draft_model_config=/tmp/draft.json",
                "training.attention_backend=flex_attention",
                "training.fsdp_strategy=FULL_SHARD",
                "training.training_num_nodes=1",
                "training.training_num_gpus_per_node=4",
                "training.micro_batch_size=1",
                "training.max_seq_length=12",
                "training.num_train_steps=null",
                "training.num_epochs=2",
                "training.dflash_num_anchors=4",
                "training.dflash_block_size=2",
                "training.dflash_num_target_layers=2",
                "training.dflash_packing=true",
                "dataset.min_loss_tokens=4",
                "dataset.shuffle_dataset=false",
                f"output_dir={self.root / 'output'}",
                *overrides,
            ]
        )
        return config_to_flat_args(config)

    def test_plan_no_drop_truncate_or_cross_epoch_merge(self):
        plan = self.fetcher().plan
        self.assertEqual(plan.rows_for_epoch(0), [[0, 1], [2], [3], [4]])
        self.assertEqual(plan.row_budget(2), 8)
        self.assertEqual(plan.row_budget(1.5), 6)
        for shuffle in (False, True):
            plan.shuffle = shuffle
            for epoch in range(5):
                rows = plan.rows_for_epoch(epoch)
                self.assertEqual(sorted(i for row in rows for i in row), list(range(5)))
                self.assertTrue(
                    all(sum(plan.records[i]["seq_len"] for i in row) <= 12 for row in rows)
                )
            self.assertEqual(
                plan.row_budget(3), sum(len(plan.rows_for_epoch(i)) for i in range(3))
            )

    def test_explicit_row_plan_replays_given_rows_per_rank(self):
        plan = [["2", "1"], ["3"], ["0", "4"]]
        plan_path = self.root / "row_plan.json"
        plan_path.write_text(json.dumps({"rows": plan}))
        args = self.args(f"offline.row_plan_path={plan_path}")
        self.assertEqual(load_row_plan(args), plan)
        fetchers = [
            OfflineBatchFetcher(
                self.dataset,
                rank=rank,
                world_size=2,
                seed=42,
                max_length=12,
                hidden_size=8,
                aux_layer_ids=[1],
                final_layer_id=3,
                vocab_size=16,
                min_loss_tokens=4,
                shuffle=False,
                packing=True,
                row_plan=load_row_plan(args),
            )
            for rank in range(2)
        ]
        records = fetchers[0].records
        index = {record["id"]: i for i, record in enumerate(records)}
        self.assertEqual(
            fetchers[0].plan.rows_for_epoch(0), [[index[s] for s in row] for row in plan]
        )
        self.assertEqual(fetchers[0].plan.row_budget(1), 3)
        self.assertEqual(next(fetchers[0])["input_ids"][0, :12].tolist(), [3] * 8 + [2] * 4)
        self.assertEqual(next(fetchers[1])["input_ids"][0, :12].tolist(), [4] * 12)
        self.assertEqual(next(fetchers[0])["input_ids"][0, :12].tolist(), [1] * 8 + [5] * 4)
        for bad, message in (
            ([["0", "9"]], "not in the training split"),
            ([["1"], ["1"]], "repeats"),
            ([["0", "3"]], "overflows"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                OfflineRowPlan(records, max_length=12, packing=True, shuffle=False, row_plan=bad)
        with self.assertRaisesRegex(ValueError, "shuffle_dataset=false"):
            OfflineRowPlan(records, max_length=12, packing=True, shuffle=True, row_plan=plan)
        with self.assertRaisesRegex(ValueError, "one sample each"):
            OfflineRowPlan(
                records, max_length=13, packing=False, shuffle=False, row_plan=[["1", "4"]]
            )

    def test_full_length_accepted_overlength_rejected(self):
        self.assertEqual(self.fetcher().plan.rows_for_epoch(0)[2], [3])
        with self.assertRaisesRegex(ValueError, "do not crop"):
            OfflineRowPlan([{"seq_len": 13}], max_length=12, packing=True)
        with self.assertRaisesRegex(ValueError, "overflows"):
            self.fetcher().load_row(self.dataset.records[:3])

    def test_exact_collator_output_and_doc_local_positions(self):
        fetcher = self.fetcher()
        actual = next(fetcher)
        expected = DFlashPackingCollator(12)(self.samples[:2])
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value)
        self.assertEqual(actual["ctx_doc_ids"].tolist(), [[0] * 8 + [1] * 4])
        self.assertEqual(actual["base_position_ids"].tolist(), [list(range(8)) + list(range(4))])
        second = next(fetcher)
        self.assertEqual(second["ctx_doc_ids"].tolist(), [[0] * 8 + [-1] * 4])
        self.assertEqual(int(second["loss_mask"].sum()), 8)
        torch.testing.assert_close(
            second["last_hidden_states"][:, :8], self.samples[2]["last_hidden_states"]
        )

    def test_packed_features_match_collating_whole_samples(self):
        from angelspec.data.positions import prepare_sample_positions

        records = self.dataset.records
        for family, collator, aux in (
            ("dflash", DFlashPackingCollator, [1]),
            ("mtp", MTPPackingCollator, []),
        ):
            fetcher = OfflineBatchFetcher(
                self.dataset,
                rank=0,
                world_size=1,
                seed=42,
                max_length=12,
                hidden_size=8,
                aux_layer_ids=aux,
                final_layer_id=3,
                vocab_size=16,
                min_loss_tokens=4,
                shuffle=False,
                packing=True,
                model_family=family,
            )
            # A full row and one with four padding positions.
            for row in ([records[0], records[1]], [records[1], records[4]]):
                samples = []
                for record in row:
                    sample = self.dataset.load(record, fetcher.layer_ids)
                    layers = sample.pop("layers")
                    sample["last_hidden_states"] = layers[3]
                    if aux:
                        sample["hidden_states"] = torch.cat([layers[a] for a in aux], dim=-1)
                    samples.append(prepare_sample_positions(sample, None))
                expected = collator(12)(samples)
                actual = fetcher.load_row(row)
                self.assertEqual(set(actual), set(expected))
                for key, value in expected.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(actual[key], value), (family, key))
                    else:
                        self.assertEqual(actual[key], value, (family, key))

    def test_dp_rows_and_sp_peers_are_deterministic(self):
        serial = self.fetcher(shuffle=True)
        expected = [next(serial)["input_ids"] for _ in range(18)]
        ranks = [self.fetcher(rank=r, world=3, shuffle=True) for r in range(3)]
        for i in range(18):
            torch.testing.assert_close(next(ranks[i % 3])["input_ids"], expected[i])
        peer = self.fetcher(rank=1, world=3, shuffle=True)
        peer.seek(9)
        torch.testing.assert_close(next(peer)["input_ids"], expected[10])
        ranks[1].seek(9)
        torch.testing.assert_close(next(ranks[1])["input_ids"], expected[10])

    def test_sp_forward_keeps_global_docs_and_gathers_correct_teacher_rows(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        batch = next(self.fetcher())
        trainer = object.__new__(DFlashTrainer)
        trainer.args = self.args(
            "training.attention_backend=usp",
            "training.sp_ulysses_size=2",
            "training.usp_local_shard=true",
        )
        trainer.num_target_layers = 2
        trainer.final_layer_from_last_hidden = True
        trainer._distill_enabled = True
        trainer.l1_loss_alpha = 0.0
        trainer.block_size = 2
        trainer.score_engine = None
        trainer.target_lm_head = SimpleNamespace(norm=torch.nn.RMSNorm(8))
        trainer.target_lm_head_weight = torch.randn(16, 8)
        trainer._sp_rank = lambda: 1
        trainer.dflash = SimpleNamespace(_extra_distill_needed=lambda: False)
        trainer._sample_usp_anchors = mock.Mock(
            return_value=(
                torch.tensor([[1, 9]]),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[9]]),
            )
        )
        trainer.model = mock.Mock(
            return_value=(
                torch.tensor(1.0),
                torch.tensor(0.0),
                torch.zeros(2),
                torch.zeros(2),
                torch.ones(2),
                {},
            )
        )
        device = torch.device
        with mock.patch("torch.device", side_effect=lambda *a, **k: device("cpu")):
            trainer._forward(batch)
        values = trainer.model.call_args.kwargs
        self.assertTrue(values["context_pre_sharded"])
        self.assertIsNone(values["last_hidden_states"])
        torch.testing.assert_close(values["ctx_doc_ids"], batch["ctx_doc_ids"])
        torch.testing.assert_close(values["base_position_ids"], batch["base_position_ids"])
        torch.testing.assert_close(values["hidden_states_list"][0], batch["hidden_states"][:, 6:])
        torch.testing.assert_close(
            values["hidden_states_list"][1], batch["last_hidden_states"][:, 6:]
        )
        torch.testing.assert_close(
            values["teacher_hidden_at_labels"], batch["last_hidden_states"][:, 8:10].unsqueeze(1)
        )

    def test_prefetch_packing_streams_rows_not_whole_steps(self):
        raw = self.fetcher()
        prefetched = prefetch_offline(raw, 1)
        self.assertFalse(prefetched.packing_enabled)
        self.assertEqual(prefetched._queue.maxsize, 1)
        reference = self.fetcher()
        try:
            for count in (2, 3):
                for actual in prefetched.iter_step_batches(count):
                    torch.testing.assert_close(actual["input_ids"], next(reference)["input_ids"])
        finally:
            prefetched.close()
            prefetched._thread.join(timeout=5)
        self.assertFalse(prefetched._thread.is_alive())

    def test_anchor_and_attention_masks_isolate_documents(self):
        batch = next(self.fetcher())
        model = DFlashModel(DFlashDraftModel(make_config()), block_size=2, num_anchors=16)
        anchors, keep = model._sample_anchor_positions(
            12, batch["loss_mask"], torch.device("cpu"), ctx_doc_ids=batch["ctx_doc_ids"]
        )
        docs = batch["ctx_doc_ids"]
        selected = anchors[keep]
        self.assertNotIn(7, selected.tolist())
        self.assertTrue(torch.equal(docs[0, selected], docs[0, selected + 1]))
        mask = _create_dflash_mask_mod(
            torch.tensor([[9]]),
            torch.tensor([[True]]),
            12,
            2,
            ctx_doc_ids=docs,
            causal=True,
        )

        def visible(q, k):
            return bool(mask(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(k)))

        self.assertFalse(visible(0, 6))
        self.assertTrue(visible(0, 8))
        self.assertFalse(visible(0, 9))
        self.assertFalse(visible(0, 13))
        self.assertTrue(visible(1, 12))

    def test_weighted_rows_match_global_supervised_weighting(self):
        lengths = torch.tensor([12, 8, 12, 4])
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        total = lengths.sum()
        distributed_loss = (
            sum(
                sum(
                    losses[i] * token_weighted_loss_scale(lengths[i], total, 2)
                    for i in range(rank, 4, 2)
                )
                for rank in range(2)
            )
            / 2
        )
        expected = (losses * lengths).sum() / total
        torch.testing.assert_close(distributed_loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(distributed_loss, losses)[0], lengths.float() / total
        )

    def _run_packed_training_loop(self, weighted):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer.args = self.args(
            f"training.dflash_packing_token_weighted_loss={str(weighted).lower()}"
        )
        trainer.model = torch.nn.Linear(1, 1, bias=False)
        trainer.model.weight.data.fill_(1)
        trainer.sp_group = None
        trainer.dp_group = None
        trainer.global_step = 0
        trainer._enable_perf_metrics = False
        trainer.prof = SimpleNamespace(iterate_train_actor=lambda rows: rows)
        trainer.data_fetcher = prefetch_offline(self.fetcher(), 1)
        trainer._maybe_dump = mock.Mock()
        trainer._aggregate_metrics = lambda values, step, grad_norm: {}
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        updates = []

        def update():
            updates.append(float(trainer.model.weight.grad.item()))
            optimizer.step()
            return 0.0

        trainer.optimizer = SimpleNamespace(step=update)
        row_scales = []

        def train_step(batch, accumulation_steps, **kwargs):
            scale = batch.get("_loss_scale", 1 / accumulation_steps)
            row_scales.append(float(scale))
            loss = trainer.model.weight.sum() * batch["input_ids"][0, 0].float()
            (loss * scale).backward()
            return {}

        trainer._train_step = train_step
        device = torch.device
        try:
            with (
                mock.patch("torch.cuda.Event"),
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch("torch.device", side_effect=lambda *a, **k: device("cpu")),
                mock.patch("torch.distributed.all_reduce"),
                mock.patch("torch.distributed.get_world_size", return_value=1),
            ):
                metrics = trainer._train_core_from_queue(step=0, num_batches=2)
        finally:
            trainer.data_fetcher.close()
            trainer.data_fetcher._thread.join(timeout=5)
        return trainer, metrics, row_scales, updates

    def test_existing_training_loop_consumes_packed_rows_and_weights(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                trainer, metrics, row_scales, updates = self._run_packed_training_loop(weighted)
                expected_scales = [0.6, 0.4] if weighted else [0.5, 0.5]
                torch.testing.assert_close(torch.tensor(row_scales), torch.tensor(expected_scales))
                self.assertAlmostEqual(updates[0], 1.8 if weighted else 2.0, places=5)
                self.assertEqual(trainer.global_step, 1)
                self.assertEqual(metrics["data/seq_tokens"], 20)
                self.assertAlmostEqual(metrics["data/fill_ratio"], 20 / 24)

    def test_packing_and_usp_config_contracts(self):
        validate_offline_args(self.args())
        usp = self.args(
            "training.attention_backend=usp",
            "training.sp_ulysses_size=2",
            "training.usp_local_shard=true",
        )
        validate_offline_args(usp)
        self.assertEqual(usp.dp_size, 2)
        self.assertEqual(usp.global_batch_size, 2)
        for override, message in [
            ("training.sp_ring_size=2", "sp_ring_size"),
            ("training.usp_local_shard=false", "usp_local_shard"),
            ("training.sp_leader_mooncake_fetch=true", "sp_leader"),
        ]:
            cfg = self.args(
                "training.attention_backend=usp",
                "training.sp_ulysses_size=2",
                "training.usp_local_shard=true",
                override,
            )
            with self.assertRaises((ValueError, NotImplementedError), msg=message):
                validate_offline_args(cfg)
        with self.assertRaisesRegex(ValueError, "requires dflash_packing"):
            validate_offline_args(
                self.args(
                    "training.dflash_packing=false",
                    "training.dflash_packing_token_weighted_loss=true",
                )
            )
        with self.assertRaisesRegex(ValueError, "packing_min_fill_ratio"):
            validate_offline_args(self.args("training.packing_min_fill_ratio=0.8"))
        from angelspec.training.offline import configure_model

        args = self.args(
            "training.attention_backend=usp",
            "training.sp_ulysses_size=2",
            "training.usp_local_shard=true",
            "training.dflash_num_anchors=3",
        )
        validate_offline_args(args)
        with self.assertRaisesRegex(ValueError, "divisible"):
            configure_model(
                args,
                self.dataset,
                make_config(),
                {"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4},
            )

    def test_driver_counts_packed_rows_and_uses_dp_not_world(self):
        args = self.args(
            "training.attention_backend=usp",
            "training.sp_ulysses_size=2",
            "training.usp_local_shard=true",
        )
        actors = [mock.Mock() for _ in range(4)]
        group = mock.Mock()
        group._actor_handlers = actors
        group.train_from_queue.return_value = [{"train/loss": 1.0}] * 4
        with (
            mock.patch(
                "angelspec.training.offline.verify_frozen_target",
                return_value={"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4},
            ),
            mock.patch("angelspec.AutoDraftModelConfig.from_file", return_value=make_config()),
            mock.patch(
                "angelspec.ray.placement_group.create_placement_groups",
                return_value={"training": ("pg", [], [])},
            ) as placement,
            mock.patch("angelspec.ray.placement_group.allocate_train_group", return_value=group),
            mock.patch("ray.get", side_effect=lambda value: value),
            mock.patch("ray.kill"),
            mock.patch("ray.util.placement_group.remove_placement_group"),
        ):
            run_offline_training(args)
        effective = placement.call_args.args[0]
        self.assertEqual(effective.dp_size, 2)
        self.assertEqual(effective.global_batch_size, 2)
        self.assertEqual(effective.num_train_steps, 4)
        self.assertEqual(group.train_from_queue.call_count, 4)

    def test_eval_packs_selected_validation_records_and_restores_rng(self):
        fetcher = self.fetcher()
        fetcher.dataset.records = [
            {**record, "split": "validation"} for record in self.dataset.records
        ]
        args = self.args("offline.eval_samples=2")
        args.seed = 99
        batches = []

        def forward(batch):
            batches.append(batch)
            torch.rand(3)
            return {"loss": 1}

        trainer = SimpleNamespace(
            args=args,
            _offline_fetcher=fetcher,
            model=torch.nn.Linear(1, 1),
            eval_forward=forward,
            _aggregate_eval_metrics=lambda values: {"rows": len(values)},
        )
        before = torch.get_rng_state().clone()
        fork_rng = torch.random.fork_rng
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.random.fork_rng", side_effect=lambda **kwargs: fork_rng(devices=[])),
        ):
            self.assertEqual(evaluate_offline(trainer), {"rows": 1})
        self.assertTrue(trainer.model.training)
        torch.testing.assert_close(before, torch.get_rng_state())
        self.assertEqual(batches[0]["ctx_doc_ids"].tolist(), [[0] * 8 + [1] * 4])

    def test_tracking_uses_existing_backends(self):
        writer = mock.Mock()
        args = SimpleNamespace(use_wandb=False, use_swanlab=False, use_tensorboard=True)
        with mock.patch("angelspec.utils.logging.get_tb_writer", return_value=writer):
            log_offline_metrics(args, {"train/loss": 1.0, "nonnumeric": []}, 3)
        writer.add_scalar.assert_any_call("train/loss", 1.0, 3)
        writer.add_scalar.assert_any_call("train/step", 3, 3)
        self.assertEqual(writer.add_scalar.call_count, 2)

    def test_nonpacked_usp_is_allowed_without_inference_engine(self):
        args = self.args(
            "training.dflash_packing=false",
            "training.attention_backend=usp",
            "training.sp_ulysses_size=2",
            "training.usp_local_shard=true",
        )
        self.assertEqual(args.inference_engine_type, "hf")
        validate_offline_args(args)
        self.assertEqual(args.dp_size, 2)
        args.offline_manifest_path = None
        from angelspec.train_entry import _validate_usp_args

        with self.assertRaisesRegex(ValueError, "inference_engine_type"):
            _validate_usp_args(args)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FlexAttention parity requires GPU")
    def test_cuda_packed_vs_individual_loss_and_gradients(self):
        torch.manual_seed(7)
        # Compiled flex attention on CUDA needs query/key and value dims of at least 16.
        draft = DFlashDraftModel(
            make_config(qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16)
        ).cuda()
        draft.freeze_embedding()
        reference = DFlashModel(
            draft,
            block_size=2,
            num_anchors=1,
            causal_attn=True,
            ce_loss_alpha=0.0,
            e2e_tv_loss_weight=1.0,
        ).cuda()
        packed = copy.deepcopy(reference)
        packed.num_anchors = 2
        head = torch.randn(16, 8, device="cuda")
        norm = torch.nn.RMSNorm(8).cuda().requires_grad_(False)

        def forward(model, batch, anchors):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            return model(
                input_ids=batch["input_ids"],
                loss_mask=batch["loss_mask"],
                hidden_states_list=[batch["hidden_states"], batch["last_hidden_states"]],
                last_hidden_states=batch["last_hidden_states"],
                lm_head_weight=head,
                target_norm=norm,
                ctx_doc_ids=batch.get("ctx_doc_ids"),
                base_position_ids=batch.get("base_position_ids"),
                injected_anchors=torch.tensor([anchors], device="cuda"),
                injected_keep_mask=torch.ones(1, len(anchors), dtype=torch.bool, device="cuda"),
            )[0]

        separate = sum(forward(reference, s, [1]) for s in self.samples[:2]) / 2
        combined = forward(packed, next(self.fetcher()), [1, 9])
        torch.testing.assert_close(combined, separate, rtol=1e-4, atol=1e-5)
        separate.backward()
        combined.backward()
        for (_, left), (_, right) in zip(reference.named_parameters(), packed.named_parameters()):
            if left.requires_grad:
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
