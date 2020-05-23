def read_txt(filepath: str) -> str:
    with open(filepath) as f:
        return f.read()
