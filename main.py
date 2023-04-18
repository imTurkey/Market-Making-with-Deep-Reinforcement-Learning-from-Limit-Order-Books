import os
import argparse
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import pyrallis
from dataclasses import asdict, dataclass

from tensorforce.environments import Environment

from environment.env_discrete import EnvDiscrete
from environment.env_continuous import EnvContinuous
from agent.tensorforce_agent import get_dueling_dqn_agent, get_ppo_agent
from network.network import *


@dataclass
class TrainConfig:
    # Experiment
    code: str = '000001'
    device: str = "cpu"
    latency: int = 1
    time_window: int = 50
    log: bool = False
    exp_name: str = ''
    # Agent
    agent_type: str = 'ppo' # ppo/dueling dqn
    learning_rate: int = 1e-4
    horizon: int = 1
    env_type: str = 'continuous' # continuous/discrete
    load: bool = False
    agent_load_dir: str = ''
    save: bool = False,
    agent_save_dir: str = ''
    # Ablation
    wo_pretrain: bool = False
    wo_attnlob: bool = False
    wo_lob_state: bool = False
    wo_market_state: bool = False
    wo_dampened_pnl: bool = False
    wo_matched_pnl: bool = False
    wo_inv_punish: bool = False


def init_env(day, config):
    if config['env_type'] == 'continuous':
        env = EnvContinuous
    elif config['env_type'] == 'discrete':
        env = EnvDiscrete
        
    environment = env(
        code=config['code'], 
        day=day,
        latency=config['latency'],
        T=config['time_window'],
        # state ablation
        wo_lob_state=config['wo_lob_state'],
        wo_market_state=config['wo_market_state'],
        # reward ablation
        wo_dampened_pnl=config['wo_dampened_pnl'],
        wo_matched_pnl=config['wo_matched_pnl'],
        wo_inv_punish=config['wo_inv_punish'],
        # exp setting
        experiment_name=config['exp_name'], 
        log=config['log'],
        )
    return environment

def init_agent(environment, config):
    kwargs=dict()
    if config['agent_type'] == 'dueling_dqn':
        get_agent = get_dueling_dqn_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']
    elif config['agent_type'] == 'ppo':
        get_agent = get_ppo_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']

    if config['wo_pretrain']:
        print("Ablation: pretrain")
        lob_model = get_lob_model(64,config['time_window'])
        lob_model.compute_output_shape = compute_output_shape
    else:
        pretrain_model_dir = f'./ckpt/pretrain_model_' + config['code']
        model = get_lob_model(64,config['time_window'])
        model.compute_output_shape = compute_output_shape
        model_pretrain = get_pretrain_model(model,config['time_window'])
        checkpoint_filepath = pretrain_model_dir + '/weights'
        model_pretrain.load_weights(checkpoint_filepath)
        lob_model = model_pretrain.layers[1] 

    if config['wo_attnlob']:
        print("Ablation: attnlob")
        lob_model = get_fclob_model(64,config['time_window'])

    model = get_model(
        lob_model,
        config['time_window'],
        with_lob_state= not config['wo_lob_state'],
        with_market_state= not config['wo_market_state']
        )
    agent = get_agent(model, environment=environment, max_episode_timesteps=1000, device=config['device'], **kwargs)

    if config['load']:
        model = keras.models.load_model(keras_model_dir)
        model.layers[1].compute_output_shape = compute_output_shape
        agent = get_agent(model, environment=environment, max_episode_timesteps=1000, device=config['device'], **kwargs)
        agent.restore(config['agent_load_dir'], filename='cppo', format='numpy')
    
    return agent

def train_a_day(environment, agent, train_result):
    num_episodes = len(environment.orderbook)//num_step_per_episode
    data_collector = list()
    for idx in tqdm(range(num_episodes)):
        episode_states = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        terminal = False
        while not terminal:
            episode_states.append(states)
            actions = agent.act(states=states, independent=True)
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        data_collector.append([episode_states, episode_actions, episode_terminal, episode_reward])

        agent.experience(
            states=episode_states, 
            actions=episode_actions, 
            terminal=episode_terminal,
            reward=episode_reward
        )

        agent.update()
        
        save_episode_result(environment, train_result)

    return episode_states, episode_actions, episode_reward

def test_a_day(environment, agent, test_result):
    num_episodes = len(environment.orderbook)//num_step_per_episode
    for idx in tqdm(range(num_episodes)):

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        terminal = False
        while not terminal:
            actions = agent.act(
                states=states, independent=True
            )
            states, terminal, reward = environment.execute(actions=actions)
        
        save_episode_result(environment, test_result)

def train(agent, train_result, config):
    for day in train_days:
        environment = init_env(day, config)
        train_a_day(environment, agent, train_result)

def test(agent, test_result, config):
    for day in test_days:
        environment = init_env(day, config)
        test_a_day(environment, agent, test_result)

def save_episode_result(environment, test_result):
    res_dict = environment.get_final_result()
    date = environment.day
    idx = environment.episode_idx
    
    test_result.loc[date+'_'+str(idx)] = [res_dict['pnl'], res_dict['nd_pnl'], res_dict['avg_abs_position'], res_dict['profit_ratio'], res_dict['volume']]

def gather_test_results(test_result):
    day_list = list(test_result.index)
    for i in range(len(day_list)):
        day_list[i] = day_list[i][:10]
    day_list = set(day_list)
    gathered_results = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    for day in day_list:
        result = test_result[test_result.index.str.contains(day)]
        pnl = result.PnL.sum()
        nd_pnl = result['ND-PnL'].sum()
        ap = result.average_position.mean()
        volume = (result.PnL/result.profit_ratio).sum()
        pr = pnl/volume
        gathered_results.loc[day] = [pnl,nd_pnl,ap,pr,volume]
    gathered_results=gathered_results.sort_index()
    return gathered_results

def save_agent(agent, config):
    # save agent network
    agent.model.policy.network.keras_model.save(keras_model_dir)
    # Save agent
    agent.save(config['agent_save_dir'], filename=agent, format='numpy')

@pyrallis.wrap()
def main(config: TrainConfig):
    config = asdict(config)

    environment = init_env(train_days[0], config)
    agent = init_agent(environment, config)

    train_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    for _ in range(n_train_loop):
        train(agent, train_result, config)
        if config['save']:
            save_agent(agent, config)

    test_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    test(agent, test_result, config)
    daily_test_results = gather_test_results(test_result)

if __name__ == '__main__':
    train_days=['20191101', '20191104', '20191105', '20191106', '20191107', '20191108', '20191111', '20191112']
    test_days=['20191113', '20191114', '20191115', '20191118', '20191119', '20191120',
            '20191121', '20191122', '20191125', '20191126', '20191127', '20191128', '20191129']
    num_step_per_episode = 2000
    n_train_loop = 5

    main()
