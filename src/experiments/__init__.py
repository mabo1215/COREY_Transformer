def main(*args, **kwargs):
    from .run_entropy_guided_experiments import main as _main

    return _main(*args, **kwargs)

__all__ = ["main"]
