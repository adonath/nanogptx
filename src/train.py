from dataclasses import dataclass

from utils import Config


@dataclass
class GlobalConfig(Config):
    """Training configuration"""

    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        True  # if True, always save a checkpoint after each eval
    )
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"  # 'run' + str(time.time())

    # data
    dataset = "openwebtext"
    gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
    batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size

    # model
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "float32"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


@dataclass
class GPTTrainer:
    """GPT trainer"""

    lr: float = 0.001
    batch_size: int = 8
    epoches: int = 10
    out_dir: str = "out"

    def train(self, x, moodel):
        pass
