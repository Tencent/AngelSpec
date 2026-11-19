_hf_configs_registered = False


def register_extra_hf_configs() -> None:
    global _hf_configs_registered
    if _hf_configs_registered:
        return

    try:
        from transformers import AutoConfig

        from angelspec.models.target.configuration_hy_v3 import HYV3Config

        AutoConfig.register("hy_v3", HYV3Config, exist_ok=True)
    except Exception:
        pass

    # transformers releases before Hy4 flash support lack its config; vLLM's resolves the
    # multimodal checkpoint to its VL config, whose text_config keeps model_type "hy_v4_flash".
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "hy_v4_flash" not in CONFIG_MAPPING:
            from vllm.models.hy_v4_flash.configs import (
                HYV4FlashConfig,
                HYV4FlashVLConfig,
            )

            AutoConfig.register("hy_v4_flash", HYV4FlashConfig, exist_ok=True)
            AutoConfig.register("hy_v4_flash_vl", HYV4FlashVLConfig, exist_ok=True)
    except Exception:
        pass

    _hf_configs_registered = True
