import os
import nni
import gym
import json
import argparse

import numpy as np

from agents import DQNAgent, DDQNAgent

str2bool = lambda x: x.lower() == 'true'
# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--agent', type=str, default='DQN')
parser.add_argument('--nni', type=str2bool, default=False)
parser.add_argument('--objective', type=str, default='test_score')
args = parser.parse_args()


def run_dqn(env, params='params/dqn_params.json', show_plot=False):
    if not isinstance(params, dict):
        with open(params, 'r') as f:
            params = json.load(f)

    dqn_agent = DQNAgent(env=env, **params)
    trained_episodes, is_trained = dqn_agent.train()
    dqn_agent.plot_scores(show=show_plot)
    score = dqn_agent.test(n_test_episodes=100)

    return dqn_agent, trained_episodes, score


def run_ddqn(env, params='params/ddqn_params.json', show_plot=False):
    if not isinstance(params, dict):
        with open(params, 'r') as f:
            params = json.load(f)

    ddqn_agent = DDQNAgent(env=env, **params)
    trained_episodes, is_trained = ddqn_agent.train()
    ddqn_agent.plot_scores(show=show_plot)
    score = ddqn_agent.test(n_test_episodes=100)

    return ddqn_agent, trained_episodes, score


def main():
    env = gym.make('LunarLanderContinuous-v2')
    params = nni.get_next_parameter() if args.nni else \
        f'params/{"dqn" if args.agent.upper() == "DQN" else "ddqn"}_params.json'

    show_plots = not args.nni
    agent, trained_episodes, test_score = run_dqn(env, params, show_plots) if args.agent.upper() == 'DQN' else run_ddqn(
        env,
        params,
        show_plots)

    if args.nni:
        nni.report_final_result(test_score if args.objective == 'test_score' else trained_episodes)


if __name__ == '__main__':
    main()
