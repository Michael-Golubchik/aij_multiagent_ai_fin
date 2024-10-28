import os
import sys
import time
import warnings
from random import shuffle
from typing import Any, Dict, List, Tuple

import pytest
from aij_multiagent_rl.agents import BaseAgent, RandomAgent

# Тест на окружении реального тест, иначе на тренировочном окружении
REAL_TEST = False
if REAL_TEST:
    from aij_multiagent_rl.env import AijMultiagentEnv
else:
    from aij_multiagent_rl_train.env import AijMultiagentEnv

from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import joblib

CONFIG_PATH = 'tests/test_config.yaml'


def sample_rollouts(
    n_rollouts: int,
    env: AijMultiagentEnv,
    agents: Dict[str, BaseAgent]
) -> Tuple[List[List[Dict[str, Any]]], float]:
    rollouts = []
    action_times = 0
    print('n_rollouts:', n_rollouts)
    
    for _ in range(n_rollouts):
        rollout = []
        for agent in agents.values():
            agent.reset_state()
        observations, infos = env.reset()
        done = False
        
        plt.imshow(env.render())
        plt.axis('off')  # Отключение осей координат
        plt.pause(0.3)  # Задержка в секундах между отображением картинок
        #plt.pause(7)  # Задержка в секундах между отображением картинок
        
        i = 0
        while not done:
            i += 1
            start = time.perf_counter()
            actions = {name: agent.get_action(observation=observations[name])
                       for name, agent in agents.items() if name in env.agents}
            end = time.perf_counter()
            action_times += (end-start)
            if REAL_TEST:
                next_observations, rewards, terminations, truncations, next_infos = env.step(actions)
            else:
                next_observations, rewards, syn_rewards, terminations, truncations, next_infos = env.step(actions)
            transition = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'rewards': rewards,
                'terminations': terminations,
                'truncations': truncations
            }
            observations = next_observations
            done = all(truncations.values()) or all(terminations.values())
            rollout.append(transition)
            
            # if not REAL_TEST:
            if 1==1:
                if i % 2 == 0:
                    plt.imshow(env.render())
                    plt.axis('off')  # Отключение осей координат
                    plt.show(block=False)
                    plt.pause(0.3)  # Задержка в секундах между отображением картинок
                    #plt.pause(5)  # Задержка в секундах между отображением картинок
                    # joblib.dump(rollout, './data/rollout_show.pkl')
                
        rollouts.append(rollout)
        print('end')
        # plt.show()
    joblib.dump(rollouts, './data/rollouts.pkl')
    action_time = action_times / (sum([len(e) for e in rollouts]) * 8)
    return rollouts, action_time


class RandomLoopedPop:

    def __init__(self, options):
        self.options = options
        self._new_shuffled_options()

    def _new_shuffled_options(self):
        self.shuffled_options = self.options.copy()
        shuffle(self.shuffled_options)

    def pop(self):
        try:
            out = self.shuffled_options.pop()
        except IndexError:
            self._new_shuffled_options()
            out = self.shuffled_options.pop()
        return out


@pytest.fixture
def config() -> DictConfig:
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture
def submission_agents(config: DictConfig) -> Dict[str, BaseAgent]:
    sys.path.insert(1, config.submission_dir)
    from model import get_agent
    agents_dir = os.path.join(config.submission_dir, 'agents')
    loaded_agents = {}
    for artifact in os.listdir(agents_dir):
        if not artifact.startswith('.'):
            artifact_dir = os.path.join(agents_dir, artifact)
            agent_config = OmegaConf.load(
                os.path.join(artifact_dir, 'agent_config.yaml'))
            agent = get_agent(agent_config)
            agent.load(artifact_dir)
            # agent.model.eval()
            loaded_agents[artifact] = agent
    return loaded_agents


@pytest.fixture
def env():
    if REAL_TEST:
        return AijMultiagentEnv()
    else:
        # return AijMultiagentEnv(stage = 0, is_one_agent = False)
        # return AijMultiagentEnv(stage = 0, is_one_agent = True)
        # return AijMultiagentEnv(stage = 1, is_one_agent = True)
        return AijMultiagentEnv(stage = 4, is_one_agent = False)
        # return AijMultiagentEnv(stage = 1, is_one_agent = True)


def test_agents_selfplay(
    config: DictConfig,
    submission_agents: Dict[str, BaseAgent],
    env: AijMultiagentEnv
):
    """Test agents self-play

    Test that agents may be used for sampling
    actions from `AijMultiagentEnv` simulator and
    therefore follow required API.

    We also test here for agents sampling performance
    in order to match 100min. time constraint in testing system.
    Obviously, testing machine will have different setup from
    the one these tests will be run on (see main contest info at:
    https://dsworks.ru/champ/multiagent-ai). However, if you have
    a GPU on your local machine, you possibly should treat
    performance warning more seriously.

    Args:
        config: tests config
        submission_agents: dictionary with initialized user
            agents and their names (names are taken from
            subdirectory name)
        env: AijMultiagentEnv simulator

    Raises:
        UserWarning: If average get action time by agents exceeds
            `config.wall_time_threshold` milliseconds
    """
    # Assign agents with valid keys from environment
    rlp = RandomLoopedPop(options=list(submission_agents.keys()))
    agents = {}
    for name in env.possible_agents:
        agent_key = rlp.pop()
        agents[name] = submission_agents[agent_key]
    # Run simulation
    _, acs_time = sample_rollouts(
        n_rollouts=config.test_episodes_num,
        env=env,
        agents=agents
    )
    tt = config.wall_time_threshold
    if acs_time > tt:
        warnings.warn(
            f"""Mean `get_action()` wall time is greater than {tt} sec ({acs_time} sec), which may be too slow"""
        )
