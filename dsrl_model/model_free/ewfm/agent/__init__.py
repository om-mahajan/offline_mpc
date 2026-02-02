import gym
from agent.sac import SAC
from agent.softq import SoftQ


def make_agent(env, args):
    obs_dim = env.observation_space.shape[-1]


    print('--> Using SAC agent')
    action_dim = env.action_space.shape[-1]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    # TODO: Simplify logic
    args.agent.obs_dim = obs_dim
    args.agent.action_dim = action_dim
    agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent
