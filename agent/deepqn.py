import torch
from .tools import Memory
from .networks import Network as Dqn
from random import random
import torch
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent:
    def __init__(self,
                json_file,
                action_space,
                obs_dim,
                load = False, file_path = "dqnsavep.pth"):
        
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hparams      = json_file["config"]["hyperparameters"]
        self._memory       = Memory(100000, obs_dim, self._hparams["a"], 0.0001)
        self._action_space = action_space

        self._batch        = self._hparams["batch"]
        self._epsilon      = self._hparams["e"]
        self._epsilon_min  = self._hparams["epsilon_min"]
        self._loss_param   = self._hparams["b"]
        self._lp_increase  = self._hparams["B"]
        self._steps        = 0

        self._decay        = self._epsilon_min**(self._hparams["epsilon_treshold"]**(-1))
        self._file = file_path

        self.dqn        = Dqn(json_file).to(self.device)
        self.target_dqn = Dqn(json_file).to(self.device)
        self._opt       = optim.Adam(self.dqn.parameters(),
                               lr=self._hparams["alpha"])
        self._optdecay  = torch.optim.lr_scheduler.ExponentialLR(optimizer=self._opt,
                                                                 gamma=self._hparams["lrdecay"])

        if load:
            self.dqn.load_state_dict(torch.load(load))
            self.target_dqn.load_state_dict(torch.load(load))

    def apply_decay(self):
        self._epsilon *= self._decay if self._epsilon > self._epsilon_min else 1

    def choose_action(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device, dtype=torch.float)
        rand  = random()
        if rand < self._epsilon:
            return 0, self._action_space.sample()
        else:
            value, index = self.dqn(state).max(0)
            return value.item(), index.item()

    def copy_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def save_on_memory(self, s, a, r, s2, d):
        self._memory.push(s, a, r, s2, d)

    def train(self):
        self._steps += 1
        if len(self._memory) < self._batch:
            return -float("inf")

        batch = self._memory.sample(self._batch)

        s  = batch[0]
        a  = batch[1]
        r  = batch[2]
        ss = batch[3]
        d  = batch[4]
        P  = batch[5]
        i  = batch[6]

        q_eval = self.dqn(s).gather(1,a.unsqueeze(1)).squeeze(1)
        q_next = self.target_dqn(ss).max(1)[0].detach()
        target = r + self._hparams["gamma"]*q_next*(1 - d)
        N      = len(self._memory)
        w      = (N * P)**(-self._loss_param)
        w      = w/w.max()
        self._loss_param *= 1 + self._lp_increase if self._loss_param < 1 else 1

        loss = F.smooth_l1_loss(q_eval, target, reduction="none")
        for index, weight in zip(i, loss):
            self._memory.update_priority(index, abs(weight.detach().item()))

        weighted_loss = loss * w.detach()
        final_loss    = torch.mean(weighted_loss)
        self._opt.zero_grad()
        final_loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self._opt.step()
        if self._hparams["lrdecay"]**self._steps > self._hparams["lrmin"]:
            self._optdecay.step()
        return final_loss

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def memory(self):
        return self._memory