from agents.PPO_agent import PPOAgent, Transition
# from TargetFinderRobo import TargetFinderRobo
from TFL1 import TargetFinderRobo




# Outside of class, RL Training Loop Implementation

env = TargetFinderRobo()

agent = PPOAgent(number_of_inputs=env.observation_space.shape[0],
                 number_of_actor_outputs=env.action_space.n)

# Load the pre-trained agent
# agent.load('models/2023-04-30_21') # Good model

solved = False
episode_count = 0
episode_limit = 5000



while not solved and episode_count < episode_limit:
    observation = env.reset()
    env.episode_score = 0

    done = False
    step = 0
    while not done:
        step += 1
        # select an action based on state
        selected_action, action_prob = agent.work(observation, type_="selectAction")

        # take the action
        new_obs, reward, done, info = env.step(selected_action)
        
        # Save the transtion in memory for training
        trans = Transition(observation, selected_action, action_prob, reward, new_obs)
        agent.store_transition(trans)

        if done:
            agent.train_step(batch_size=step+1)
            solved = env.solved()
            trans = Transition(observation, selected_action, action_prob, reward, new_obs)
            agent.store_transition(trans)
            break

        observation = new_obs

    print(f"Episode: {episode_count}\tscore: {env.episode_score}")
    agent.writer.add_scalar('episode_reward', env.episode_score, episode_count)
    episode_count += 1


# Save
agent.save(f'models/')

if not solved:
    print("Task is not solved, deploying for testing...")
else:
    print("Agent is ready for deployment. Deploying now...")
    agent.save('models/TFL1', best=True)

observation = env.reset()
env.episode_score = 0.0
# while True:
#     selected_action, action_prob = agent.work(observation, type_="selectActionMax")
#     observation, _, done, _ = env.step(selected_action)
#     if done:
#         observation = env.reset()

        