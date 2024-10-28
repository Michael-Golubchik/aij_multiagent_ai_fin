import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from aij_multiagent_rl.agents import BaseAgent
from omegaconf import DictConfig
from torch import nn
from utils.networks import QCNN
from utils.utils import from_numpy, get_device, to_numpy
import keyboard


class DQNAgent(BaseAgent, nn.Module):
    def __init__(
        self,
        model,
        device,
        eval_mode,
        warmup_steps=20000,
        eps_start=0.2,
        eps_decay=0.995,
        eps_decay_every=5000,
        acs_dim=9,
        seed=None
    ):
        super(DQNAgent, self).__init__()
        self.device = device
        self.warmup_steps = warmup_steps
        self.eval_mode = eval_mode
        self.steps_made = 0
        self.n_target_updates = 0
        self.current_eps = eps_start
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_decay_every = eps_decay_every
        self.acs_dim = acs_dim
        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.target_model.to(device)
        if seed is None:
            seed = np.random.randint(0, int(1e6), 1)
        self.rng = np.random.default_rng(seed)
        # Нет ввода с клавиатуры
        self.no_keyboard = False

    def reset_state(self) -> None:
        pass

    def load(self, ckpt_dir: str) -> None:
        self.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "module.pth"),
                map_location=get_device()
            )
        )

    def save(self, ckpt_dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(ckpt_dir, 'module.pth'))

    def _eps(self):
        if self.steps_made < self.warmup_steps:
            return 1.
        else:
            if self.steps_made % self.eps_decay_every == 0:
                self.current_eps *= self.eps_decay
            return self.current_eps

    def _get_action(self, observation: Dict[str, np.ndarray]) -> int:
        image = from_numpy(self.device, np.expand_dims(observation['image'], 0))
        proprio = from_numpy(self.device, np.expand_dims(observation['proprio'], 0))
        qvals = self.model.forward(image=image, proprio=proprio)
        qvals = to_numpy(qvals)
        if self.no_keyboard:
            return np.argmax(qvals)
        else:
            # Если требуется нажатие клавиатуры
            # Ожидаем нажатия цифры
            digit_pressed = self.wait_for_digit()
            if digit_pressed == 9:
                # Отключаем работу с клавиатурой
                self.no_keyboard = True
                return np.argmax(qvals)
            else:
                return digit_pressed

    def wait_for_digit(self):
        while True:
            event = keyboard.read_event()  # Читаем событие клавиатуры
            if event.event_type == 'down' and event.name.isdigit():  # Проверяем, что это нажатие цифры
                return np.int64(event.name)  # Возвращаем нажатую цифру

    def update_target_network(self, tau: float) -> None:
        self.n_target_updates += 1
        for target_param, param in zip(
                self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.model.forward(image, proprio)

    @torch.no_grad()
    def forward_target(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.target_model.forward(image, proprio)

    def get_action(self, observation: Dict[str, np.ndarray]) -> int:
        if not self.eval_mode:
            self.steps_made += 1
        eps = self._eps()
        u = self.rng.uniform(0, 1, 1).item()
        if u < eps:
            acs = self.rng.integers(0, self.acs_dim, 1).item()
        else:
            acs = self._get_action(observation=observation)
        return acs


def get_agent(config: DictConfig) -> BaseAgent:
    agent = DQNAgent(
        device=get_device(),
        model=QCNN(in_channels=config.in_channels, acs_dim=config.acs_dim),
        eval_mode=config.eval_mode,
        warmup_steps=config.warmup_steps,
        eps_start=config.eps_start,
        eps_decay=config.eps_decay,
        eps_decay_every=config.eps_decay_every,
        acs_dim=config.acs_dim,
        seed=config.seed
    )
    agent.steps_made = config.ckpt_steps_made
    agent.current_eps = config.ckpt_eps
    return agent
