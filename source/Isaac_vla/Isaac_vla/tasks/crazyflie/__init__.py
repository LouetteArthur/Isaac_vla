"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .crazyflie_cfg import CrazyflieEnv, CrazyflieEnvCfg
##
# Register Gym environments.
##


gym.register(
    id="Isaac-crazyflie",
    entry_point="Isaac_vla.tasks.crazyflie.crazyflie_cfg:CrazyflieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrazyflieEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrazyfliePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cnn_cfg.yaml",
    },
)