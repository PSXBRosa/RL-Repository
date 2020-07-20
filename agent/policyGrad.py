from .networks import Network
from torch.distributions import Categorical
import torch
from torch.optim import Adam

class PGAgent:
    def __init__(self,
                json_file,
                action_space,
                obs_dim,
                load = False,
                file_path = "dqnsavep.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hparams       = json_file["config"]["hyperparameters"]
        self._gamma   = hparams["gamma"]
        self._alpha   = hparams["alpha"]
        self._obs_dim = obs_dim
        self._as      = action_space
        self._network = Network(json_file).to(self.device)
        self._optmin  = Adam(self._network.parameters(),
                             self._alpha)
        self.reset_memory()

    def reset_memory(self):
        self._memory ={"states"  : torch.zeros((500,self._obs_dim)),
                       "actions" : torch.zeros(500),
                       "rewards" : torch.zeros(500).double(),
                       "sstates" : torch.zeros((500,self._obs_dim)),
                       "dones"   : torch.zeros(500)}

    def __setitem__(self, index, exp):
        s,a,r,ss,d = exp
        self._memory["states" ][index] = torch.tensor(s)
        self._memory["actions"][index] = torch.tensor(a)
        self._memory["rewards"][index] = torch.tensor(r)
        self._memory["sstates"][index] = torch.tensor(ss)
        self._memory["dones"  ][index] = torch.tensor(d)
        
    def __call__(self, x):
        x = torch.tensor(x).float()
        logits = self._network(x.unsqueeze(0)).to(self.device)
        probs = Categorical(logits)
        actions = probs.sample().cpu().detach().numpy()
        return actions

    def _loss(self, logits, actions, rewards):
        probs     = Categorical(logits)
        log_probs = probs.log_prob(actions)
        cum_sum   = self._cummulativesum(rewards)

        policy_loss = -(log_probs*cum_sum).mean()
        return policy_loss

    def _cummulativesum(self, rewards):
        n        = len(rewards)
        discout  = self._gamma ** torch.arange(1,n+1, dtype = torch.float64)
        returns  = torch.flip(rewards, [0]) * discout
        returns  = torch.cumsum(returns, dim=0)
        returns  = torch.flip(returns, [0]) 
        mean     = returns.mean()
        stdev    = torch.std(returns)
        returns  =  (returns - mean)/stdev
        return returns


    def train(self, steps):
        states  = self._memory["states"][:steps]
        actions = self._memory["actions"][:steps]
        rewards = self._memory["rewards"][:steps]
        logits  = self._network(states.reshape(-1,self._obs_dim))
        
        loss = self._loss(logits, actions, rewards)
        self._optmin.zero_grad()
        loss.backward()
        self._optmin.step()
        return loss