import json
from pathlib import Path

import yaml

from angelspec.data.positions import PositionRecipe, position_recipe_from_config
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.models.draft.dfly import DFlyConfig
from angelspec.models.draft.dfly2 import DFly2Config, DFly2DraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.training.position_config import configure_draft_positions
from tests.test_mtp_mrope import ROPE

ROOT = Path(__file__).resolve().parents[1]


def test_hy_v4_flash_dfly_config_dispatches_to_dfly():
    config = AutoDraftModelConfig.from_file(
        ROOT / "angelspec/config/dfly_hy_v4_flash_draft_config.json"
    )
    assert isinstance(config, DFlyConfig)
    assert config.model_arch == "dfly"


def test_hy_v4_multimodal_recipes_reconstruct_hy4_positions():
    # Online VL capture refuses a draft config without the Hy4 position recipe.
    recipes = sorted((ROOT / "configs").glob("vllm_hy_v4_*.yaml"))
    assert recipes
    for path in recipes:
        recipe = yaml.safe_load(path.read_text())
        if not recipe["inference"]["vllm"].get("enable_multimodal"):
            continue
        target = {
            "model_type": "hy_v4_flash" if "_flash_" in path.name else "hy_v4_vl",
            "vision_config": {"use_image_newline_token": None},
            "text_config": {
                "qk_rope_head_dim": 64,
                "layer_types": ["deepseek_sparse_attention"],
                "rope_parameters": {"deepseek_sparse_attention": dict(ROPE)},
            },
        }
        draft = AutoDraftModelConfig.from_file(ROOT / recipe["model"]["draft_model_config"])
        configure_draft_positions(draft, target)
        assert position_recipe_from_config(draft) == PositionRecipe(), path.name


def test_hy_v4_flash_dfly2_config_keeps_its_name_through_a_save(tmp_path):
    config = AutoDraftModelConfig.from_file(
        ROOT / "angelspec/config/dfly2_hy_v4_flash_draft_config.json"
    )
    assert isinstance(config, DFly2Config)
    assert isinstance(config, DSparkConfig)
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert (saved["model_type"], saved["architectures"]) == ("dfly2", ["DFly2DraftModel"])
    assert isinstance(AutoDraftModelConfig.from_file(str(tmp_path / "config.json")), DFly2Config)


def test_dspark_trainer_builds_each_name_as_its_own_class():
    from angelspec.training.dspark_trainer import DSparkTrainer

    tiny = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        num_target_layers=1,
        target_hidden_size=64,
        target_num_hidden_layers=4,
        mask_token_id=31,
        markov_rank=4,
    )
    build = DSparkTrainer._build_draft_model
    assert type(build(None, DFly2Config(**tiny))) is DFly2DraftModel
    assert type(build(None, DSparkConfig(**tiny))) is DSparkDraftModel


def test_dfly2_defaults_to_no_markov_head_unlike_dspark():
    assert (DFly2Config().markov_rank, DFly2Config().confidence_head_with_markov) == (0, False)
    assert (DSparkConfig().markov_rank, DSparkConfig().confidence_head_with_markov) == (256, True)
    config = AutoDraftModelConfig.from_file(
        ROOT / "angelspec/config/dfly2_hy_v4_flash_draft_config.json"
    )
    assert (config.markov_rank, config.confidence_head_with_markov) == (0, False)


def test_dspark_config_has_dspark_model_arch():
    config = AutoDraftModelConfig.from_file(
        ROOT / "angelspec/config/dspark_qwen3_8b_draft_config.json"
    )
    assert isinstance(config, DSparkConfig)
    assert config.model_arch == "dspark"


def test_legacy_draft_config_name_resolves():
    config = AutoDraftModelConfig.from_file("dflash_draft_config.json")
    assert config.model_arch == "dflash"
