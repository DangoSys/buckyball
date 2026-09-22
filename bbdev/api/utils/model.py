def model_layout_name(model: str) -> str:
    aliases = {
        "lenet": "LeNet",
        "mobilenet": "MobileNetV3",
        "resnet": "ResNet18",
        "yolo": "YOLO26",
        "bert": "Bert",
        "qwen3": "Qwen3",
        "gemma4": "Gemma4",
        "deepseekr1": "DeepSeekR1",
        "llama2": "llama2",
        "stable-diffusion": "StableDiffusion",
        "whisper": "Whisper",
        "buddynext": "BuddyNext",
    }
    try:
        return aliases[model.lower()]
    except KeyError as error:
        raise ValueError(f"unknown model: {model}") from error
