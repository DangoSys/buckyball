def model_layout_name(model: str) -> str:
    aliases = {
        "lenet": "lenet",
        "mobilenet": "mobilenet-v3-small",
        "resnet": "resnet18",
        "yolo": "yolo26n",
        "bert": "bert",
        "qwen3": "qwen3-8b",
        "gemma4": "gemma4-e2b-it",
        "deepseekr1": "deepseek-r1-0528-qwen3-8b",
        "llama2": "llama2",
        "stable-diffusion": "stable-diffusion",
        "whisper": "whisper",
    }
    try:
        return aliases[model.lower()]
    except KeyError as error:
        raise ValueError(f"unknown model: {model}") from error
