def req_arg(body: dict, name: str):
    return body.get(name) or body.get(name.replace("_", "-"))
