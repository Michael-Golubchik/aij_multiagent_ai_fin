#!/usr/bin/env python
# coding: utf-8

# # Бейзлайн для задачи AIJ Multi-Agent AI
# 
# Данный ноутбук содержит реализацию [VDN](https://arxiv.org/abs/1706.05296) - кооперативного 
# мульти-агентного алгоритма обучения с подкреплением. VDN основан на предпосылке
# о линейном разложении общей награды агентов, таким образом, общая награда
# всех агентов представлена в виде суммы индивидуальных наград.
# Несмотря на то, что данная предпосылка ограничивает класс обучаемых стратегий
# только кооперативными вариантами, VDN все же является хорошим бейзлайном для 
# многих задач мульти-агентного обучения с подкреплением.
# 
# Данный бейзлайн позволяет получить целевую метрику (Mean Focal Score) около
# 42 при ее сабмите в тестовую систему (Случайная политика, для сравнения, 
# получает ~4).
# 
# Главным результатом работы ноутбука будет создание директории `submission_vdn`, которую 
# необходимо запаковать в .zip архив и отправить в тестирующую систему.
# 
# Мы рекомендуем запускать тесты на своих решениях, прежде чем отправлять их в систему.

# In[ ]:


import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from aij_multiagent_rl.agents import BaseAgent, RandomAgent
from aij_multiagent_rl_train.env import AijMultiagentEnv
from math import ceil
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import time
# import logging
from collections import defaultdict
import joblib
import random

import sys
import shutil
import zipfile

library_dir = 'library'
submission_dir = 'submission_vdn'
sys.path.insert(1, library_dir)
# sys.path.insert(1, submission_dir)
from model import DQNAgent, get_agent
#from model_with_keys import DQNAgent, get_agent

# if not show:
#     # Если в режиме "show" (игра с клавиатуры) каталоги не созадем
#     create_dirs()
#     from model import DQNAgent, get_agent
# else:
#     # Если в режиме show. то импортируем агента который управляется с клавиатуры
#     from model_with_keys import DQNAgent, get_agent

from utils.utils import get_device, from_numpy
from utils.networks import QCNN

DEBUG = False

config = DictConfig({
    'n_stages': 6,  # Число этапов тренировки
    'warmup_steps': 20000,
    #'warmup_steps': 0,
    'eps_start': 0.2,
    'eps_decay': 0.996,
    #'eps_decay': 0.999,
    'eps_decay_every': 1000,
    'acs_dim': 9,
    'batch_size': 32,
    # 'batch_size': 128,
    'update_every': 4,
    'buffer_size': 100000,
    'initial_batch_episodes': 20,
    #'initial_batch_episodes': 1,
    'learning_rate': 0.00005,
    #'learning_rate': 0.000005,
    'gamma': 0.99,
    #'target_updates_freq': 15,
    'target_updates_freq': 1,
    'episodes_per_iter': 2,
    #'episodes_per_iter': 10,
    'iter_per_save': 10,
    'n_iters_0': 300,
    'n_iters_1': 200, 
    'n_iters_2': 200,
    'n_iters_3': 200,
    'n_iters_4': 161,
    'n_iters_5': 872,
    'tau': 0.005,
    'output_dir': f'{submission_dir}/agents',
    'in_channels': 3
})

# Контекстный менеджер для замера времени
class Timer:
    total_times = defaultdict(float)  # Словарь для суммирования времени по частям
    call_counts = defaultdict(int)  # Словарь для подсчёта вызовов по частям
    log_data = []  # Сюда будем сохранять информацию о времени выполнения

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_time = time.time() - self.start_time
        # log_entry = f"{self.name} занял {self.elapsed_time:.4f} секунд"
        # logging.info(log_entry)
        Timer.log_data.append((self.name, self.elapsed_time))  # Запоминаем
        Timer.total_times[self.name] += self.elapsed_time  # Добавляем к общему времени для каждой части
        Timer.call_counts[self.name] += 1  # Увеличиваем счётчик вызовов


# Создаем корректную структуру сабмишена и другие необходимые для обучения каталоги
def create_dirs():
    
    # Делаем каталоги для сабмита
    required_dirs = [
        f'{submission_dir}/agents',
        f'{submission_dir}/utils',
        f'{submission_dir}/best_agent/agent_0',
        f'{submission_dir}/last_agent/agent_0',
    ]
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Копируем в каталоги для сабмита необходимые модули
    shutil.copy(f'{library_dir}/model.py', submission_dir)
    shutil.copy(f'{library_dir}/utils/utils.py', f'{submission_dir}/utils')
    shutil.copy(f'{library_dir}/utils/networks.py', f'{submission_dir}/utils')
    
    # Делаем каталоги для обучения
    for i in range(config.n_stages):
        stage_dir = f'data/stage_{i}'
        if not os.path.exists(stage_dir):
            os.makedirs(stage_dir)
    other_agents_dir = f'data/other_agents/agent_0'
    if not os.path.exists(other_agents_dir):
        os.makedirs(other_agents_dir)
    agents_history_dir = f'data/agents_history'
    if not os.path.exists(agents_history_dir):
        os.makedirs(agents_history_dir)


# # Функция сэмплирования из среды
# 
# Далее реализован параллельный сэмплинг данных из среды `AijMultiagentEnv` при помощи нескольких агентов. В ходе работы функции вызываются следующие методы:
# 
# 1) `reset_state()` - перезагрузка внутреннего состояния агента с началом эпизода
# 2) `get_action()` - получение действия из композитного наблюдения
def sample_rollouts(
    n_rollouts: int,
    env: AijMultiagentEnv,
    my_agents: Dict[str, BaseAgent],
    other_agents: Dict[str, BaseAgent],
    verbose: Optional[bool] = False,
    show = False,
    it = 0,
) -> List[List[Dict[str, Any]]]:
    rollouts = []
    agents = get_agents_mix(my_agents=my_agents, other_agents=other_agents, env=env, show=show)
    
    for _ in tqdm(range(n_rollouts), disable=not verbose):
        rollout = []
        with Timer("rollouts_1"):
            for agent in agents.values():
                agent.reset_state()
        with Timer("rollouts_2"):
            all_observations, all_infos = env.reset()
            done = False
        all_dops = {name: agent.get_dop()
                for name, agent in agents.items() if name in env.agents}
        all_prev_proprios = {name: np.concatenate(agent.get_prev_proprios())
                             for name, agent in agents.items() if name in env.agents}
        i = 0
        with Timer("rollouts_3"):
            while not done:
                # Отображение текущего состояния
                if show:
                # if show and i % 100 == 0:
                    plt.imshow(env.render())
                    plt.axis('off')  # Отключение осей координат
                    plt.draw()  # Рисуем изображение в окне
                    plt.pause(0.01)  # Короткая пауза для обновления экрана
                with Timer("rollouts_3_1"):
                    # if i < 10:
                    #     print('dops:', dops)
                    # Дополнительные данные агента до действия
                    all_actions = {name: agent.get_action(observation=all_observations[name])
                               for name, agent in agents.items() if name in env.agents}
                    # Дополнительные данные агента после действия
                    all_next_dops = {name: agent.get_dop()
                                     for name, agent in agents.items() if name in env.agents}
                    # Дополнительные данные агента после действия
                    all_next_prev_proprios = {name: np.concatenate(agent.get_prev_proprios())
                                              for name, agent in agents.items() if name in env.agents}

                # with Timer("rollouts_3_2"):
                all_next_observations, all_rewards, all_syn_rewards, all_terminations, all_truncations, all_next_infos = env.step(all_actions)
                observations, next_observations, prev_proprios, next_prev_proprios, dops, next_dops = {}, {}, {}, {}, {}, {}
                actions, rewards, syn_rewards, terminations, truncations = {}, {}, {}, {}, {}
                my_agent_i = 0
                for name, agent in agents.items():
                    if not agent.eval_mode:
                        observations[f'agent_{my_agent_i}'] = all_observations[name]
                        next_observations[f'agent_{my_agent_i}'] = all_next_observations[name]
                        dops[f'agent_{my_agent_i}'] = all_dops[name]
                        next_dops[f'agent_{my_agent_i}'] = all_next_dops[name]
                        prev_proprios[f'agent_{my_agent_i}'] = all_prev_proprios[name]
                        next_prev_proprios[f'agent_{my_agent_i}'] = all_next_prev_proprios[name]
                        actions[f'agent_{my_agent_i}'] = all_actions[name]
                        rewards[f'agent_{my_agent_i}'] = all_rewards[name]
                        syn_rewards[f'agent_{my_agent_i}'] = all_syn_rewards[name]
                        terminations[f'agent_{my_agent_i}'] = all_terminations[name]
                        truncations[f'agent_{my_agent_i}'] = all_truncations[name]
                        my_agent_i += 1

                transition = {
                    'observations': observations,
                    'next_observations': next_observations,
                    'dops': dops,
                    'next_dops': next_dops,
                    'prev_proprios': prev_proprios,
                    'next_prev_proprios': next_prev_proprios,
                    'actions': actions,
                    'rewards': rewards,
                    'syn_rewards': syn_rewards,
                    'terminations': terminations,
                    'truncations': truncations
                }
                all_observations = all_next_observations
                all_dops = all_next_dops
                all_prev_proprios = all_next_prev_proprios
                # dops = next_dops
                done = all(truncations.values()) or all(terminations.values())
                rollout.append(transition)
                # Сохраняем шаг если включено отображение
                if show:
                # if show and i % 100 == 0:
                    plt.imshow(env.render())
                    plt.axis('off')  # Отключение осей координат
                    plt.draw()  # Рисуем изображение в окне
                    plt.pause(0.01)  # Короткая пауза для обновления экрана
                    proprio_data = rollout[-1]['observations']['agent_0']['proprio']
                    rounded_proprio = [round(val, 2) for val in proprio_data]
                    # print(rounded_proprio)
                    # joblib.dump(rollout, './data/rollout_key.pkl')
                # if i > 1:
                #     dop_info = rollout[1]['dops']['agent_0']
                # if i > 1 and i < 10:
                #     print('dop_info:', dop_info)
                i += 1
        rollouts.append(rollout)
        if it % 100 == 0:
            joblib.dump(rollouts, './data/rollouts_cur.pkl')
    return rollouts


# # Сэмплирование эпизодов при помощи случайных агентов
def get_mean_agent_return(batch, env, reward_type='rewards', agents_num=8):
    mean_rews = []
    for path in batch:
        ep_tot_rew = [sum(t[reward_type].values()) for t in path]
        ep_tot_rew = sum(ep_tot_rew)
        # mean_rews.append(ep_tot_rew / 8)
        mean_rews.append(ep_tot_rew / agents_num)
    return np.mean(mean_rews)

# # Создаем буфер данных
# 
# Создаем простой буфер данных для хранения эпизодов симуляции
class ReplayBuffer(Dataset):

    def __init__(
        self,
        n_transitions: int
    ):
        self.rollouts = []
        self.n_transitions = n_transitions
        self.lengths = None

    def add_batch(self, rollouts):
        self.rollouts.extend(rollouts)
        self._evict()
        self.lengths = [len(r) for r in self.rollouts]

    def _evict(self) -> None:
        while len(self) > self.n_transitions:
            self.rollouts.pop(0)

    def __len__(self):
        if len(self.rollouts) == 0:
            return 0
        else:
            return sum([len(r) for r in self.rollouts])

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
        c_lengths = np.cumsum(self.lengths)
        r_ind = np.argwhere(c_lengths > idx).min()
        r_ind_last = 0
        if r_ind > 0:
            r_ind_last = c_lengths[r_ind - 1]
        t_ind = idx - r_ind_last
        transition = self.rollouts[r_ind][t_ind]
        item = {
            **transition,
            'rollout_index': r_ind,
            'transition_index': t_ind
        }
        return item


def collate_fn(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_data = {k: {} for k in data[0].keys()}
    for a in data[0]['observations'].keys():
        collated_data['observations'][a] = {}
        collated_data['observations'][a]['image'] = np.array(
            [d['observations'][a]['image'] for d in data])
        collated_data['observations'][a]['proprio'] = np.array(
            [d['observations'][a]['proprio'] for d in data])
        collated_data['next_observations'][a] = {}
        collated_data['next_observations'][a]['image'] = np.array(
            [d['next_observations'][a]['image'] for d in data])
        collated_data['next_observations'][a]['proprio'] = np.array(
            [d['next_observations'][a]['proprio'] for d in data])
        collated_data['dops'][a] = np.array([d['dops'][a] for d in data])
        collated_data['next_dops'][a] = np.array([d['next_dops'][a] for d in data])
        collated_data['prev_proprios'][a] = np.array([d['prev_proprios'][a] for d in data])
        collated_data['next_prev_proprios'][a] = np.array([d['next_prev_proprios'][a] for d in data])
        collated_data['actions'][a] = np.array([d['actions'][a] for d in data])
        collated_data['rewards'][a] = np.array([d['rewards'][a] for d in data])
        collated_data['syn_rewards'][a] = np.array([d['syn_rewards'][a] for d in data])
        collated_data['terminations'][a] = np.array([d['terminations'][a] for d in data])
        collated_data['truncations'][a] = np.array([d['truncations'][a] for d in data])
    collated_data['rollout_index'] = np.array([d['rollout_index'] for d in data])
    collated_data['transition_index'] = np.array([d['transition_index'] for d in data])
    return collated_data


class VDNTrainer(nn.Module):
    def __init__(
        self,
        agents: Dict[str, DQNAgent],
        learning_rate: float,
        gamma: float = 0.99,
        td_criterion=F.smooth_l1_loss,
        tau: float = 0.005
    ):
        super(VDNTrainer, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.td_criterion = td_criterion
        self.tau = tau
        self.n_updates = 0
        self.last_logs = {}
        self.i = 0

        # Set agents
        self.agents = nn.ModuleDict(agents)
        self.devices = {n: a.device for n, a in self.agents.items()}

        # Define optimizer
        self.optimizer = optim.Adam(
            params=(self._get_params()),
            lr=self.learning_rate
        )

    def _get_params(self):
        params = []
        ids = []
        # with Timer("_get_params"):
        for a in self.agents.values():
            model_id = id(a.model)
            if model_id not in ids:
                params.extend(list(a.model.parameters()))
                ids.append(model_id)
        return params

    def update_target_networks(self) -> None:
        # with Timer("update_target_networks"):
        for a in self.agents.values():
            a.update_target_network(tau=self.tau)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        proprio: Dict[str, torch.Tensor],
        prev_proprios: Dict[str, torch.Tensor],
        dop: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = {}
        # with Timer("forward"):
        for name, agent in self.agents.items():
            qvals = agent.forward(images[name], proprio[name], prev_proprios[name], dop[name])
            output[name] = qvals
        return output

    @torch.no_grad()
    def forward_target(
        self,
        images: Dict[str, torch.Tensor],
        proprio: Dict[str, torch.Tensor],
        prev_proprios: Dict[str, torch.Tensor],
        dop: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = {}
        # with Timer("forward_target"):
        for name, agent in self.agents.items():
            qvals = agent.forward_target(images[name], proprio[name], prev_proprios[name], dop[name])
            output[name] = qvals
        return output

    def save(self, dir: str, config: dict, history_dir: str) -> None:
        shutil.rmtree(dir)
        # with Timer("save"):
        # for name, agent in self.agents.items():
        #     agent_dir = os.path.join(dir, name)
        #     if not os.path.exists(agent_dir):
        #         os.makedirs(agent_dir)
        #     agent.save(ckpt_dir=agent_dir)
        #     with open(os.path.join(agent_dir, 'agent_config.yaml'), 'w') as outfile:
        #         yaml.dump(config, outfile, default_flow_style=False)
        
        # name = 'agent_0'
        # agent = self.agents[name]
        # agent_dir = os.path.join(dir, name)
        # if not os.path.exists(agent_dir):
        #     os.makedirs(agent_dir)
        # agent.save(ckpt_dir=agent_dir)
        # with open(os.path.join(agent_dir, 'agent_config.yaml'), 'w') as outfile:
        #     yaml.dump(config, outfile, default_flow_style=False)

        agent = self.agents['agent_0']
        for i in range(8):
            name = f'agent_{i}'
            agent_dir = os.path.join(dir, name)
            if not os.path.exists(agent_dir):
                os.makedirs(agent_dir)
            agent.save(ckpt_dir=agent_dir)
            with open(os.path.join(agent_dir, 'agent_config.yaml'), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
        
        copy_with_overwrite(os.path.join(dir, 'agent_0'), history_dir)    

    def update(self, sample: Dict[str, Any], stage) -> Dict[str, float]:

        # with Timer("update"):
        # Get device
        devs = self.devices

        # Unpack data
        obs_image = {k: from_numpy(devs[k], v['image']) 
                     for k, v in sample['observations'].items()}
        next_obs_image = {k: from_numpy(devs[k], v['image']) 
                          for k, v in sample['next_observations'].items()}
        obs_proprio = {k: from_numpy(devs[k], v['proprio']) 
                       for k, v in sample['observations'].items()}
        next_obs_proprio = {k: from_numpy(devs[k], v['proprio']) 
                            for k, v in sample['next_observations'].items()}
        dops = {k: from_numpy(devs[k], v) for k, v in sample['dops'].items()}
        next_dops = {k: from_numpy(devs[k], v) for k, v in sample['next_dops'].items()}
        prev_proprios = {k: from_numpy(devs[k], v) for k, v in sample['prev_proprios'].items()}
        next_prev_proprios = {k: from_numpy(devs[k], v) for k, v in sample['next_prev_proprios'].items()}
        # if self.i < 2:
        #     print("dops:", dops)
        #     print("next_dops:", next_dops)
        #     # print("obs_proprio:", obs_proprio)
        #     # print("next_obs_proprio:", next_obs_proprio)
        # self.i += 1
        actions = {k: from_numpy(devs[k], v) for k, v in sample['actions'].items()}
        rewards = {k: from_numpy(devs[k], v) / 10. for k, v in sample['rewards'].items()}
        syn_rewards = {k: from_numpy(devs[k], v) / 10. for k, v in sample['syn_rewards'].items()}
        terminations = {k: from_numpy(devs[k], v) for k, v in sample['terminations'].items()}

        shared_rewards = torch.cat([r.unsqueeze(-1) for r in rewards.values()], axis=-1)
        shared_rewards = shared_rewards.sum(dim=-1, keepdims=True)

        syn_shared_rewards = torch.cat([r.unsqueeze(-1) for r in syn_rewards.values()], axis=-1)
        syn_shared_rewards = syn_shared_rewards.sum(dim=-1, keepdims=True)

        # construct target q-values
        qa_tp1_target = self.forward_target(next_obs_image, next_obs_proprio, next_prev_proprios, next_dops)
        with torch.no_grad():
            qa_tp1_model = self.forward(next_obs_image, next_obs_proprio, next_prev_proprios, next_dops)

        # Select maximum value by agent and sum
        q_tp1 = []
        for name, qa_tp1_t_a in qa_tp1_target.items():
            qa_tp1_m_a = qa_tp1_model[name]
            q_tp1_a = torch.gather(qa_tp1_t_a, 1,
                                   qa_tp1_m_a.argmax(dim=1, keepdims=True))
            term = terminations[name].unsqueeze(1)
            q_tp1_a = q_tp1_a * torch.logical_not(term)
            q_tp1.append(q_tp1_a)
        q_tp1 = torch.cat(q_tp1, axis=-1).sum(dim=-1, keepdims=True)

        # Create targets
        # q_targets = shared_rewards + self.gamma * q_tp1
        # q_targets = syn_shared_rewards + self.gamma * q_tp1
        full_shared_rewards = syn_shared_rewards+shared_rewards
        if stage >= 100:
            # Смешиваем синтетический таргет с реальным
            q_targets = full_shared_rewards + self.gamma * q_tp1
        else:
            q_targets = syn_shared_rewards + self.gamma * q_tp1
        # Calculate outputs
        qa_t = self.forward(obs_image, obs_proprio, prev_proprios, dops)

        # Select qvalue by action
        q_t = []
        for name, qa_t_a in qa_t.items():
            acs_a = actions[name]
            q_t_a = torch.gather(qa_t_a, 1,
                                 acs_a.to(torch.long).unsqueeze(1))
            q_t.append(q_t_a)
        q_t = torch.cat(q_t, axis=-1).sum(dim=-1, keepdims=True)

        # compute loss
        loss = self.td_criterion(q_t, q_targets)

        # performing gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1

        return {
            'dqn_loss': loss.item(),
            'mean_q_value': q_t.mean().item(),
            'n_updates': self.n_updates,
            'mean_shared_reward': shared_rewards.mean().item(),
            'syn_mean_shared_reward': syn_shared_rewards.mean().item(),
            'full_mean_shared_rewards': full_shared_rewards.mean().item(),
            'mean_agents_steps': np.mean([a.steps_made for a in self.agents.values()]).item(),
            'mean_target_updates': np.mean([a.n_target_updates for a in self.agents.values()]).item(),
            'mean_eps': np.mean([a.current_eps for a in self.agents.values()]).item()
        }


def get_agents_mix(my_agents, other_agents, env, show):
    '''
    param: my_agents: словарь с обучаемымми агентами
    param: other_agents: словарь с другими агентами на поле
    param: env: среда выполнения агентов
    return: возвращает микс из обучаемых и других агентов
    '''
    agents = {}
    # Выбираем случайное число других агентов от 0 до максимум общее чило возмолжных агентов минус 1
    # other_agents_num = random.randint(0, len(env.possible_agents) - 1)
    # Список ключей с выбранными другими агентами
    other_agents_keys = random.sample(env.possible_agents, k=len(other_agents))
    # Индексы в списках агентов:
    other_agents_i = 0
    my_agents_i = 0
    for a in env.possible_agents:
        if a in other_agents_keys:
            agents[a] = other_agents[f'agent_{other_agents_i}']
            other_agents_i += 1
            if show:
                print(a, 'другие агенты')
        else:
            if show:
                print(a, 'мои агенты')
            agents[a] = my_agents[f'agent_{my_agents_i}']
            my_agents_i += 1
        
    return agents

def train_stage(n_iters = 0,
                stage = 0,
                start_trained_path = False,
                is_one_agent = True,
                show = False,
                other_agents_num = 0
               ):
    '''
    param: stage: номер этапа
    param: start_trained: путь из которого брать модель для дальнейщего обучения на этапе
    param: is_one_agent: True если один агент
    param: show:  True если в режиме показа обучения
    param: other_agents_num:  Чиссло других агентов, которые ранее обучены, есть на поле но не учаться.
    '''

    global config

    if show:
        '''
        Если в режиме показа меняем параметры чтобы он точно слушался клавиатуры,
        а не приблизительно и иногда подменял чем-то случайным. И отключаем прогрев, чтоб сразу брал действия у агента
        '''
        config.initial_batch_episodes = 1
        config.warmup_steps = 1
        config.eps_start = 0
            
    env = AijMultiagentEnv(stage=stage, is_one_agent = is_one_agent)
    buffer = ReplayBuffer(config.buffer_size)
    
    device = get_device()
    
    model = QCNN(in_channels=config.in_channels, acs_dim=config.acs_dim)
    
    if not show and start_trained_path != '':
        state_dict = torch.load(start_trained_path)
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(new_state_dict)
        print('load model')
    elif show:
        # Путь к последнему сохранению модуля
        cur_module_dir = os.path.join(config.output_dir, 'agent_0/module.pth')
        if show:
            print(cur_module_dir)
        state_dict = torch.load(cur_module_dir)
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(new_state_dict)
    
    # agents = {a: DQNAgent(
    #     device=device,
    #     model=model,
    #     eval_mode=False,
    #     warmup_steps=config.warmup_steps,
    #     eps_start=config.eps_start,
    #     eps_decay=config.eps_decay,
    #     eps_decay_every=config.eps_decay_every,
    #     acs_dim=config.acs_dim
    # ) for a in env.possible_agents}
    
    # Число "других" (не обучаемых агентов конкурентов)
    # other_agents_num = 0 if is_one_agent else 4
    # Число "моих" (обучаемых агентов)
    my_agents_num = len(env.possible_agents) - other_agents_num
    agents = {}
    my_agents = {}
    other_agents = {}
    for i in range(other_agents_num):
        a = f'agent_{i}'
        other_agents[a] = DQNAgent(
            device=device,
            model=QCNN(in_channels=config.in_channels, acs_dim=config.acs_dim),
            eval_mode=True,
            warmup_steps=config.warmup_steps,
            eps_start=config.eps_start,
            eps_decay=config.eps_decay,
            eps_decay_every=config.eps_decay_every,
            acs_dim=config.acs_dim
        )
        other_agents[a].load('data/other_agents/agent_0')
        other_agents[a].model.eval()
        #if show:
        other_agents[a].eps_start = 0
        other_agents[a].warmup_steps = 0
        
    for i in range(my_agents_num):
        a = f'agent_{i}'
        my_agents[a] = DQNAgent(
            device=device,
            model=model,
            eval_mode=False,
            warmup_steps=config.warmup_steps,
            eps_start=config.eps_start,
            eps_decay=config.eps_decay,
            eps_decay_every=config.eps_decay_every,
            acs_dim=config.acs_dim
        )
               
    trainer = VDNTrainer(
        agents=my_agents,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau
    )
      
    initial_batch = sample_rollouts(
        n_rollouts=config.initial_batch_episodes,
        env=env,
        my_agents=trainer.agents,
        other_agents=other_agents,
        verbose=True,
        show=show
    )
    buffer.add_batch(initial_batch)
    dataloader = DataLoader(
        dataset=buffer,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True,
    )
    print(f'Initial Buffer Size: {len(buffer)}')
    
    
    # # Запускаем обучение
    training_logs = []
    
    #Максимум средних наград между записями агентов на диск
    mean_agent_rewards_mean_max = -1000000
    
    
    for it in tqdm(range(n_iters)):
    
        batch = sample_rollouts(env=env, my_agents=trainer.agents, other_agents=other_agents,
                                n_rollouts=config.episodes_per_iter, verbose=False, show=show, it=it)
        # with Timer("Part_1_2"):
        batch_size = sum([len(e) for e in batch])
    
        # Add to buffer
        # with Timer("Part_1_3"):
        buffer.add_batch(rollouts=batch)
        # with Timer("Part_1_4"):
        data_iter = iter(dataloader)
        trainer.i = 0
    
        # with Timer("Part_2"):
        # Launch update loop
        # with Timer("Part_2_0"):
        iter_n_updates = max(1, batch_size // config.update_every)
        iter_logs = []
        for _ in range(iter_n_updates):
            try:
                # with Timer("Part_2_1"):
                sample = next(data_iter)
            except StopIteration:
                # with Timer("Part_2_2"):
                data_iter = iter(dataloader)
                sample = next(data_iter)
            with Timer("Part_2_3"):
                logs = trainer.update(sample, stage)
            if trainer.n_updates % config.target_updates_freq == 0:
                # with Timer("Part_2_4"):
                trainer.update_target_networks()
            # with Timer("Part_2_5"):
            iter_logs.append(logs)
    
        # with Timer("Part_3"):
        # Collect Logs
        mean_agent_reward = get_mean_agent_return(batch, env, reward_type='rewards', agents_num=my_agents_num)
        syn_mean_agent_reward = get_mean_agent_return(batch, env, reward_type='syn_rewards', agents_num=my_agents_num)
        mean_episode_length = batch_size / config.episodes_per_iter
        iter_logs = {
            'mean_agent_reward': mean_agent_reward,
            'syn_mean_agent_reward': syn_mean_agent_reward,
            'my_agents_num': my_agents_num,
            'mean_episode_length': mean_episode_length,
            'batch_size': batch_size,
            'iter_n_updates': iter_n_updates,
            'buffer_size_transitions': len(buffer),
            'buffer_size_episodes': len(buffer.rollouts),
            **{k: np.mean([l[k] for l in iter_logs]) for k in iter_logs[0].keys()},
        }
        training_logs.append(iter_logs)
        joblib.dump(training_logs, './data/training_logs.pkl')
    
        # with Timer("Part_4"):
        # Write artifacts
        if it > 0 and it % config.iter_per_save == 0:
            ckpt_steps_made = min([a.steps_made for a in trainer.agents.values()])
            ckpt_eps = min([a.current_eps for a in trainer.agents.values()])
            save_config = {
                **config,
                'eval_mode': True,
                'ckpt_steps_made': ckpt_steps_made,
                'ckpt_eps': ckpt_eps,
                'seed': 42
            }
            trainer.save(dir=config.output_dir,
                         config=save_config,
                         history_dir=f'data/agents_history/{it}')
            # Последние config.iter_per_save значений mean_agent_reward
            mean_agent_rewards_last = [log['mean_agent_reward'] for log in training_logs[-config.iter_per_save:]]
            
            # Среднее значение последних mean_agent_reward
            mean_agent_rewards_mean = np.mean(mean_agent_rewards_last)
            if mean_agent_rewards_mean > mean_agent_rewards_mean_max:
                # Если новое среднее лучше прошлого, то записываем агента в лучщие агенты
                mean_agent_rewards_mean_max = mean_agent_rewards_mean
                print('Новый максимум наград агентов:', mean_agent_rewards_mean_max)
                copy_with_overwrite(f'{submission_dir}/agents/agent_0', f'{submission_dir}/best_agent/agent_0') 


def copy_with_overwrite(source_dir, target_dir):
    # Проверяем, существует ли целевая директория
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Создаём целевую директорию, если её нет
    
    # Копируем содержимое с перезаписью
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def copy_stage_agents(stage):
    '''
    Сохраняет агентов указанного этапа в соответствующие каталоги
    partam: stage: этап для которого нужно скопировать агентов
    '''
    # Копируем одного первого агента чтобы продолджить обучать его модель на следующих этапах
    # shutil.copy(f'{submission_dir}/agents/agent_0/module.pth', 'data/module.pth')
    
    # Сохраняем на всякий случай всех агентов указанного этапа на будущее
    # Путь к исходному каталогу
    source_dir = f'{submission_dir}/agents'
    # Путь к целевому каталогу
    target_dir = f'data/stage_{stage}'
    
    copy_with_overwrite(source_dir, target_dir)


def zip_dir(zipf, dirname):
    '''
    Функция для добавления файлов и папок в zip-архив
    '''
    for root, _, files in os.walk(dirname):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(dirname, '..')))

def make_submit_zip():
    '''
    Создает zip файл для сабмита
    '''
    
    # Имя создаваемого zip-архива
    zip_name = f'{submission_dir}/submission.zip'
        
    # Создание zip-архива и добавление папок agents и utils, а также файла model.py
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zip_dir(zipf, os.path.join(submission_dir, 'agents'))
        zip_dir(zipf, os.path.join(submission_dir, 'utils'))
        zipf.write(os.path.join(submission_dir, 'model.py'), 'model.py')
    
    print(f'Zip-архив {zip_name} успешно создан.')


def train_main(start_trained_path = '',
               show = False):
    '''
    Проводим обучение агентов целиком, последовательно по всем этапам
    param: start_trained_path: если нулевой этап тоже брать из уже
    начавшейся обучаться ранее модели
    '''
    global config
    
    if not show:
        # Если в режиме "show" (игра с клавиатуры) каталоги не созадем
        create_dirs()

    if True:
        '''
        Нулевой этап
        Тренируем собирать мусор и кидать его соседям.
        '''
        print('Тренируем нулевой этап:')
        config.eps_start = 0.25
        config.eps_decay = 0.999
        train_stage(n_iters = config.n_iters_0,
                    stage = 0,
                    start_trained_path = start_trained_path,
                    is_one_agent = True,
                    show = show,
                    other_agents_num = 0
                   )
        copy_stage_agents(0)

    if True:
        '''
        Первый этап
        Тренируем кидать дальше.
        '''
        config.warmup_steps = 0
        config.eps_start = 0.10
        config.eps_decay = 0.999
        print('Тренируем первый этап:')
        train_stage(n_iters = config.n_iters_1,
                    stage = 1,
                    start_trained_path = 'data/stage_0/agent_0/module.pth',
                    is_one_agent = True,
                    show = show,
                    other_agents_num = 0
                   )
        copy_stage_agents(1)
    if True:
        '''
        Второй этап
        Тренируем на уборку мусора.
        '''
        print('Тренируем второй этап:')
        config.warmup_steps = 0
        config.eps_start = 0.20
        config.eps_decay = 0.999
        train_stage(n_iters = config.n_iters_2,
                    stage = 2,
                    start_trained_path = 'data/stage_1/agent_0/module.pth',
                    is_one_agent = True,
                    show = show,
                    other_agents_num = 0
                   )
        copy_stage_agents(2)
    if True:
        '''
        Третий этап
        Тренируем с экологией.
        '''
        print('Тренируем третий этап:')
        config.warmup_steps = 0
        config.eps_start = 0.10
        config.eps_decay = 0.999
        train_stage(n_iters = config.n_iters_3,
                    stage = 3,
                    start_trained_path = 'data/stage_2/agent_0/module.pth',
                    is_one_agent = True,
                    show = show,
                    other_agents_num = 0
                   )
        copy_stage_agents(3)
    if True:
        '''
        Четвертый этап
        Тренируем 8 агентов.
        '''
        print('Тренируем четвертый этап:')
        config.warmup_steps = 0
        config.eps_start = 0.10
        config.eps_decay = 0.997
        train_stage(n_iters = config.n_iters_4,
                    stage = 4,
                    start_trained_path = 'data/stage_3/agent_0/module.pth',
                    is_one_agent = False,
                    show = show,
                    other_agents_num = 0
                   )
        copy_stage_agents(4)
        
    '''
    Копируем агента с завершившегося 4-го этапа для использования
    в качестве "других" агентов на фоне которых будем дальше тренировать
    наших агентов
    '''
    source_dir = f'{submission_dir}/agents/agent_0'
    target_dir = f'data/other_agents/agent_0'
    copy_with_overwrite(source_dir, target_dir)
    
    if True:
        '''
        Пятый этап
        Тренируем 4 агента своих на фоне 4-х других из прошлого этапа.
        Добавил что на черное можно кидать близко
        '''
        print('Тренируем завершающий пятый этап:')
        config.warmup_steps = 0
        config.eps_start = 0.10
        config.eps_decay = 0.997
        train_stage(n_iters = config.n_iters_5,
                    stage = 5,
                    start_trained_path = 'data/stage_4/agent_0/module.pth',
                    is_one_agent = False,
                    show = show,
                    other_agents_num = 4
                   )
        copy_stage_agents(5)

    # Сохраняем итог. Те агенты что получились.
    # Последних тренированных агентов из каталога сабмита в каталог последнего агента:
    copy_with_overwrite(f'{submission_dir}/agents/agent_0', f'{submission_dir}/last_agent/agent_0') 
    # Очищаем каталог для сабмита
    shutil.rmtree(f'{submission_dir}/agents/')
    # Лучшего агента в каталог агентов для сабмита:
    copy_with_overwrite(f'{submission_dir}/best_agent/agent_0', f'{submission_dir}/agents/agent_0') 
    # Создаем архив для сабмита
    make_submit_zip()
    # # Вывод времени выполнения
    # Вывод общего времени выполнения и количества вызовов для каждой части
    for name in Timer.total_times:
        print(f"{name}: общее время = {Timer.total_times[name]:.4f} секунд, количество вызовов = {Timer.call_counts[name]}")