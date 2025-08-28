from enum import StrEnum


class DatasetEnum(StrEnum):
    """Dataset enum"""

    shakespeare = "shakespeare"
    openwebtext = "openwebtext"
    fineweb_10b = "fineweb-10b"
    fineweb_100b = "fineweb-100b"
    fineweb_edu_10b = "fineweb-edu-10b"
    fineweb_edu_100b = "fineweb-edu-100b"
    tinystories = "tinystories"
    pile_uncopyrighted = "pile-uncopyrighted"


class PretrainedModels(StrEnum):
    """Pretrained models"""

    resume = "resume"
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"
