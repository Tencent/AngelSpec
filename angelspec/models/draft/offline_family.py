from angelspec.models.draft.dflash import DFlashConfig, build_target_layer_ids
from angelspec.models.draft.dfly import DFlyConfig
from angelspec.models.draft.dspark import DSparkConfig, dspark_training_key


def offline_family(config):
    if isinstance(config, DSparkConfig):
        if config.model_arch not in ("dspark", "dflash"):
            raise ValueError("DSparkConfig requires model_arch dspark or dflash")
        family = "dspark"
    elif isinstance(config, DFlyConfig):
        if config.model_arch not in ("dflash", "dfly"):
            raise ValueError("DFlyConfig has an incompatible model_arch")
        family = "dfly"
    elif type(config) is DFlashConfig and config.model_arch in ("dflash", "dflare"):
        family = config.model_arch
    else:
        raise ValueError(
            "Offline features support DFlash, DFly, DFlare and DSpark, not MTP/Eagle3"
        )
    if getattr(config, "layer_arch", "dflash") != "dflash":
        raise ValueError("This branch supports only the dense dflash layer body")
    if family == "dflare" and config.attention_type != "gqa":
        raise ValueError("DFlare supports GQA only; use DFly for MLA with layer fusion")
    if family != "dflash" and config.fusion_type != "concat_fc":
        raise ValueError(f"{family} cannot use the DFlash gated_sum fusion override")
    return family


def offline_layer_ids(config):
    offline_family(config)
    layers = config.target_layer_ids
    if layers is None:
        layers = build_target_layer_ids(config.num_target_layers, config.target_num_hidden_layers)
    layers = list(layers)
    if (
        not layers
        or len(layers) != config.num_target_layers
        or len(set(layers)) != len(layers)
        or any(i < 0 or i >= config.target_num_hidden_layers for i in layers)
    ):
        raise ValueError("Invalid target layer IDs/count")
    final = config.target_num_hidden_layers - 1
    if final in layers and layers[-1] != final:
        raise ValueError("Final target layer must be last in target_layer_ids")
    return layers


def offline_training_knobs(args, config):
    family = offline_family(config)
    layers_key, anchors_key = (
        dspark_training_key(args, name) if family == "dspark" else f"dflash_{name}"
        for name in ("num_target_layers", "num_anchors")
    )
    layers = getattr(args, layers_key, None)
    if layers is None:
        layers = config.num_target_layers
        setattr(args, layers_key, layers)
    anchors = getattr(args, anchors_key)
    if layers != config.num_target_layers:
        raise ValueError(f"training.{layers_key} differs from draft config")
    if anchors < 1:
        raise ValueError(f"training.{anchors_key} must be positive")
    return family, layers, anchors
