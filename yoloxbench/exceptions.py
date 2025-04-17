class YoxError(Exception):
    """Base for all custom exceptions."""

def _fmt(msg: str, hint: str | None = None):
    return f"{msg}\nHint: {hint}" if hint else msg

class ConfigError(YoxError):
    pass

class DataError(YoxError):
    pass

class TrainError(YoxError):
    pass

class ValError(YoxError):
    pass
