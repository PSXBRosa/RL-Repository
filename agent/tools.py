from random import choices
from torch import zeros, tensor, device, cuda, int64

class Memory:
    """Class to help the deep q network to replay
    the experiences it has had. It is implemented
    with priority sampling."""
    def __init__(self, maxlen, dim, a, offset):
        self.device  = device("cuda" if cuda.is_available() else "cpu")
        self._maxlen = maxlen
        self._a      = a
        self._offset = offset
        self._memory = {
        "states"  :zeros((maxlen,dim)).float().to(self.device),
        "actions" :zeros(maxlen,dtype=int64).to(self.device),
        "rewards" :zeros(maxlen).float().to(self.device),
        "nstates" :zeros((maxlen,dim)).float().to(self.device),
        "dones"   :zeros(maxlen).float().to(self.device),
        "priority":zeros(maxlen).double().to(self.device)
        }
        self._index = 0
        self._sum = 0
        self._len   = 0

    def _iindex(self):
        """Increases the index according to the
        maximum lenght. If the index is greater 
        then the maximum lenght, it starts again."""
        self._index = (self._index + 1)%self._maxlen

    def push(self, s, a, r, s2, d):
        if self._index > 0:
            mx = max(self._memory["priority"][:self._index])
        else: mx = 1
        self._sum += mx - self._memory["priority"][self._index].item()
        self._len += 1 if len(self) < self._maxlen else 0
        self._memory["states"][self._index]   = tensor(s)
        self._memory["actions"][self._index]  = a
        self._memory["rewards"][self._index]  = r
        self._memory["nstates"][self._index]  = tensor(s2)
        self._memory["dones"][self._index]    = d

        self._memory["priority"][self._index] = mx
        self._iindex()

    def sample(self, batch):
        n = len(self)
        indexes = choices(range(n),
                          weights = self._memory["priority"].tolist()[:n],
                          k = batch)
        to_return = (self._memory["states"  ][indexes],
                     self._memory["actions" ][indexes],
                     self._memory["rewards" ][indexes],
                     self._memory["nstates" ][indexes],
                     self._memory["dones"   ][indexes],
                     self._memory["priority"][indexes]/self._sum,
                     indexes)
        return to_return

    def update_priority(self, index, newp):
        q = (newp + self._offset)**self._a
        self._sum += q - self._memory["priority"][index]
        self._memory["priority"][index] = q

    def __len__(self):
        return self._len

class ScoreTracker:
    def __init__(self, hf):
        self._buffer = []
        self._mean = 0
        self._hf = hf

    def push(self, x):
        self._buffer.append(x)
        self._mean += x
        if len(self._buffer) >= self._hf:
            removed = self._buffer.pop(0)
            self._mean -= removed

    @property
    def mean(self):
        return self._mean/len(self._buffer)