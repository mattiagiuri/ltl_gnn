def get_resource(resource: str) -> str:
    path = f'resources/{resource}'
    try:
        with open(path, 'r') as file:
            resource_text = file.read()
    except FileNotFoundError:
        with open(f'test/{path}', 'r') as file:
            resource_text = file.read()
    return resource_text
