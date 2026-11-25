_hf_configs_registered = False


def register_extra_hf_configs() -> None:
    global _hf_configs_registered
    if _hf_configs_registered:
        return

    from angelspec.plugins import load_plugins

    load_plugins()

    try:
        from transformers import AutoConfig

        from angelspec.models.target.configuration_hy_v3 import HYV3Config

        AutoConfig.register("hy_v3", HYV3Config, exist_ok=True)
    except Exception:
        pass

    # transformers ships no DeepSeek-V4.1 config; vLLM's flattens text_config onto the
    # top level, which is what the target-side readers expect.
    try:
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

        AutoConfig.register("deepseek_v41", DeepseekV41Config, exist_ok=True)
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

    # Likewise Hy4: a transformers release may know "hy_v4" but not its "hy_v4_vl" wrapper.
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "hy_v4_vl" not in CONFIG_MAPPING:
            from vllm.transformers_utils.configs.hy_v4 import HYV4Config, HYV4VLConfig

            if "hy_v4" not in CONFIG_MAPPING:
                AutoConfig.register("hy_v4", HYV4Config, exist_ok=True)
            AutoConfig.register("hy_v4_vl", HYV4VLConfig, exist_ok=True)
    except Exception:
        pass

    _hf_configs_registered = True
