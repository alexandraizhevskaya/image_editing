from pydantic import BaseModel
from typing import Optional


class TrainParams(BaseModel):
    device: str
    size: int
    batch_size: int
    n_samples: int
    output_dir: str
    lr: float
    mixing: float
    trained_weights_path: Optional[str]
    weights_path: str
    training_iterations: int
    src_class: str
    tgt_class: str
    direction_lambda: float
    global_lambda: float
    output_interval: int
    clip_model_name: str
    change_type: Optional[str]
    sample_truncation: float
    save_interval: int
    channel_multiplier: int
    checkpoint_path: str
    seed: Optional[int]
    