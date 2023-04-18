from tensorforce.agents import Agent

def get_dueling_dqn_agent(
                        network, 
                        environment=None, 
                        states=None,
                        actions=None,
                        max_episode_timesteps=None,
                        batch_size=32, 
                        learning_rate=1e-4, 
                        horizon=1, 
                        discount=0.99,
                        memory=200000, 
                        device='gpu'
                        ):
    if environment != None:
        agent = Agent.create(
        agent='dueling_dqn',
        environment=environment,
        max_episode_timesteps=max_episode_timesteps,
        network=network,
        config=dict(device=device),
        memory=memory,
        batch_size=batch_size, 
        learning_rate=learning_rate,
        horizon=horizon,
        discount=discount,
        parallel_interactions=10,
    )
    else:
        agent = Agent.create(
            agent='dueling_dqn',
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            memory=memory,
            batch_size=batch_size, 
            learning_rate=learning_rate,
            horizon=horizon,
            discount=discount,
            parallel_interactions=10,
        )
    return agent

def get_ppo_agent(
                network, 
                environment=None, 
                states=None,
                actions=None,
                max_episode_timesteps=None,
                batch_size=32, 
                learning_rate=1e-3,
                horizon=None, 
                discount=0.99,
                device='gpu'
                ):
    if environment != None:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            batch_size=batch_size, 
            learning_rate=learning_rate,
            discount=discount,
            parallel_interactions=10,
        )
    else:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device),
            batch_size=batch_size, 
            learning_rate=learning_rate,
            discount=discount,
            parallel_interactions=10,
        )

    return agent