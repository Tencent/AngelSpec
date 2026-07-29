# Customization

## New target model

To train against a target model AngelSpec has not been used with before, set
`model.target_model_path` to its path or HuggingFace ID. Models supported by the inference backend
work directly.

For a model the backend does not ship with, register it at runtime rather than patching the
installed backend. AngelSpec registers such models (and their config classes) through its runtime
patch, so both the inference engine and the trainer can load the target. If you add a new target,
make sure any trainer that reads the target config (for the frozen LM head and norm) also
registers the config class, since the trainer runs in a separate process from the engine.

## New chat template

Chat templates control how conversations are rendered into tokens and which tokens are supervised.
Register a new template by adding an entry to the chat-template registry
(`angelspec/data/template.py`), then reference it by name:

```yaml
dataset:
  chat_template: your-template-name
```

The template defines the role headers, system prompt, and end-of-turn token, and which parser to
use (for example a parser that preserves separate reasoning content). Verify a new template
produces the token IDs and loss mask you expect before launching a long run.

## New draft model

Draft models are dispatched by config type in `angelspec/models/draft/auto.py`. To add a new draft
architecture:

1. Implement the model. Autoregressive drafters implement the `Eagle3DraftModel` interface
   (`models/draft/base.py`); block-parallel drafters can follow the DFlash interface contract
   (`extract_context_feature` / `forward` / `embed_tokens`).
2. Define its config class (a `PretrainedConfig` subclass).
3. Register both in `auto.py` — add the config→model mapping in `AutoEagle3DraftModel` and the
   architecture-name→config mapping in `AutoDraftModelConfig`.
4. Add a trainer if the training math differs, and dispatch it in `trainer_actor.py` by config
   type. If it reuses the DFlash forward, attach differences through the DFlash extension hooks
   instead of copying the forward (this is how DFlare and DSpark are built).
5. Add it to the FSDP2 sharding set if it introduces large modules.

Once the draft config's `architectures` field names your model, the trainer and checkpoint tooling
build and reload the correct architecture automatically.
