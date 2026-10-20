"""
Model families supported by llama.cpp.

Each family maps to the chat template that is passed to `llama-server` via its
`--chat-template` argument. The value `None` means that the template embedded
in the GGUF model file is used (the `--jinja` template engine, which is
enabled by default in recent llama.cpp builds). This is the recommended
option because almost all GGUF releases ship with the correct chat template
in their metadata.

The built-in template names below are the ones accepted by `llama-server
--chat-template`. See
https://github.com/ggml-org/llama.cpp/tree/master/tools/server for the
authoritative list.
"""

# Model family label -> `--chat-template` value (`None` = use the template
# embedded in the GGUF).
MODEL_FAMILIES: dict[str, str | None] = {
    'Auto (template embedded in the GGUF)': None,
    'Qwen3-VL / Qwen2.5-VL / Qwen2-VL (ChatML)': 'chatml',
    'InternVL (ChatML)': 'chatml',
    'Gemma 3 / PaliGemma': 'gemma',
    'LLaVA v1.5 (Vicuna)': 'vicuna',
    'LLaVA-NeXT v1.6 (Vicuna)': 'vicuna',
    'LLaVA-NeXT v1.6 (Mistral)': 'mistral-v1',
    'LLaVA-Llama-3': 'llama3',
    'Llama 3.2 Vision': 'llama3',
    'Llama 4 (Scout / Maverick)': 'llama4',
    'Mistral Small 3.1 / 3.2, Pixtral (Tekken v3)': 'mistral-v3-tekken',
    'Mistral Medium 3 (Tekken v7)': 'mistral-v7-tekken',
    'MiniCPM-V': 'minicpm',
    'Phi-3 / Phi-3.5 Vision': 'phi3',
    'Phi-4 Multimodal': 'phi4',
    'SmolVLM': 'smolvlm',
    'GLM-4V / ChatGLM': 'chatglm4',
    'Aya Vision (Command R)': 'command-r',
    'Hunyuan-VL': 'hunyuan-vl',
    'Granite Vision': 'granite',
    'DeepSeek-OCR': 'deepseek-ocr',
    'ExaONE 4.0': 'exaone4',
    'Falcon 3': 'falcon3',
    'Grok 2 Vision': 'grok-2',
    'OpenChat': 'openchat',
    'Orion': 'orion',
    'Zephyr': 'zephyr',
    'Llama 2': 'llama2',
}

MODEL_FAMILY_LABELS = list(MODEL_FAMILIES)

DEFAULT_MODEL_FAMILY = MODEL_FAMILY_LABELS[0]


def get_chat_template(model_family: str) -> str | None:
    """
    Get the `--chat-template` value for a model family label. `None` is
    returned for unknown families and for the `Auto` option, in which case
    the template embedded in the GGUF is used.
    """
    return MODEL_FAMILIES.get(model_family)
