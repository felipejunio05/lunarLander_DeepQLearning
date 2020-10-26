from ._agent import Agent
from tensorflow import config

__all__ = ["Agent"]

gpus = config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
