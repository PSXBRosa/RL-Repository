from agent.deepqn import DQNAgent as Agent
from torch import save as save_model
import matplotlib.pyplot as plt
from gym import make
from agent.tools import ScoreTracker
import json
import pickle

with open("structures/structureCartpole.json") as data_file:
    data = data_file.read()
    print(data)
    structure = json.loads(data)

env   = make("CartPole-v1")
agent = Agent(structure, env.action_space, 4)
sc    = ScoreTracker(50)
n     = 1000
save  = 100

N_episodes = 350
rewards    = [0]*N_episodes
mean_rds   = [0]*N_episodes
ttl_steps  = 0
for i in range(1,N_episodes+1):
    steps     = 0
    ep_loss   = 0
    ep_reward = 0
    d         = False
    s         = env.reset()
    avg_value = 0
    while not d:
        steps     += 1
        ttl_steps += 1
        value, action = agent.choose_action(s)
        s2, r, d, _   = env.step(action)
        rd = r if r < 490 else r + 10
        agent.save_on_memory(s, action, rd, s2, d)
        loss  = agent.train()
        s = s2
        ep_reward += r
        avg_value += value
        ep_loss   += loss
        if ttl_steps%n == 0: agent.copy_weights()
    if i%save == 0:
        save_model(agent.dqn.state_dict(), f"BackUps/{str(i)}.pth")
        with open(f"BackUps/{str(i)}.pickle", "wb") as f:
            pickle.dump(rewards, f)
    agent.apply_decay()
    sc.push(ep_reward)
    if sc.mean >= 495:
        break
    print(f"episode {i:04d} | {int(ep_reward):03d} | {(avg_value/steps):.2f} | {steps:03d} | {agent.epsilon:.2f} | {len(agent.memory):06d} "
          f"| {(ep_loss/steps):.2f} | {sc.mean:.2f}")
    rewards[i-1]  = ep_reward
    mean_rds[i-1] = sc.mean 

with plt.xkcd():
    x = [i for i in range(N_episodes)]
    plt.plot(x, rewards,alpha=0.1)
    plt.plot(x, mean_rds,alpha=1)
    plt.show()
