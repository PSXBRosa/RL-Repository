from agent.policyGrad import PGAgent as Agent
from agent.tools import ScoreTracker
from gym import make
import json
import matplotlib.pyplot as plt

with open("structures/structurePGRAD.json") as data_file:
    data = data_file.read()
    print(data)
    structure = json.loads(data)

env   = make("CartPole-v0")
agent = Agent(structure, env.action_space, 4)
sc    = ScoreTracker(50)
save  = 100
N_episodes = 300
rewards    = [0]*N_episodes
mean_rds   = [0]*N_episodes
for i in range(1,N_episodes+1):
    
    steps     = 0
    ep_loss   = 0
    ep_reward = 0
    d         = False
    s         = env.reset()

    while not d:
        action = agent(s)
        s2, r, d, _ = env.step(action)
        agent[steps] = (s, action, r, s2, d)

        s = s2
        ep_reward += r
        steps     += 1
 
    sc.push(ep_reward)
    rewards[i-1]  = ep_reward
    mean_rds[i-1] = sc.mean
    for j in range(2):
        loss = agent.train(steps)
        if loss is not None and loss > -float("inf"):
            ep_loss   += loss
    agent.reset_memory()
    print(f"(ep : {i:03d} | rewards : {int(ep_reward):03d} | loss : {ep_loss:.2f} | mean : {sc.mean:.2f})")
    if sc.mean == 200:
        break
with plt.xkcd():
    x = [k for k in range(i)]
    plt.plot(x, rewards[:i],alpha=0.1)
    plt.plot(x, mean_rds[:i],alpha=1)
    plt.show()