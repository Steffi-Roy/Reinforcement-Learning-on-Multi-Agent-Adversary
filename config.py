from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    
    n_good_agents: int = 2          # N in simple_adversary_v3 (cooperative agents)
    max_cycles: int = 25            # Max steps per episode

    # Training 
    num_episodes: int = 10_000
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    warmup_steps: int = 2_000       # Random transitions before learning starts

    
    lr_actor: float = 1e-2
    lr_critic: float = 1e-2

    gamma: float = 0.95
    tau: float = 0.01               # Soft target-network update rate

    noise_std_start: float = 0.3
    noise_std_end: float = 0.01
    noise_decay_steps: int = 100_000

    # Network 
    hidden_dim: int = 64

    log_interval: int = 200
    eval_episodes: int = 100
    rolling_window: int = 100

    
    device: str = "cpu"

    
    #simple_adversary_v3 has exactly 1 adversary and N good agents.
    ##Todo: add scale
    use_multi_adversary: bool = False
    multi_n_good_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    multi_num_episodes: int = 3_000  # Shorter run for each configuration
