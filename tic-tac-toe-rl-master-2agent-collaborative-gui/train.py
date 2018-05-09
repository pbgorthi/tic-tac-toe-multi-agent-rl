import tensorflow as tf
import numpy as np
from collections import deque
from rl.deep_q_network import DeepQNetwork
from game import Game

# initialize game env
env = Game()

# initialize tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("logs/value_network", sess.graph)

# prepare custom tensorboard summaries
episode_reward = tf.Variable(0.)
tf.summary.scalar("Last 100 Episodes Average Episode Reward", episode_reward)
summary_vars = [episode_reward]
summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
summary_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

# define policy neural network
state_dim   = 9
num_actions = 9

summaries = tf.summary.merge_all()
q_network_p1 = DeepQNetwork("p1",
                         sess,
                         optimizer,
                         state_dim,
                         num_actions,
                         init_exp=0.6,         # initial exploration prob
                         final_exp=0.1,        # final exploration prob
                         anneal_steps=120000,  # N steps for annealing exploration
                         discount_factor=0.8)  # no need for discounting

q_network_p2 = DeepQNetwork("p2",
                         sess,
                         optimizer,
                         state_dim,
                         num_actions,
                         init_exp=0.6,         # initial exploration prob
                         final_exp=0.1,        # final exploration prob
                         anneal_steps=120000,  # N steps for annealing exploration
                         discount_factor=0.8)  # no need for discounting

# load checkpoint if there is any
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("successfully loaded checkpoint")

# how many episodes to train
training_episodes = 200000

# store episodes history
p1_episode_history = deque(maxlen=100)
p2_episode_history = deque(maxlen=100)

p1_lost = 0
p1_won = 0
p2_lost = 0
p2_won = 0
draw = 0
#cheated = 0

# start training
reward_p1 = 0.0
reward_p2= 0.0
for i_episode in range(training_episodes):
  state = np.array(env.reset())
  done = False
  while done==False:
    state_p1 = state
    action_p1 = q_network_p1.eGreedyAction(state[np.newaxis,:])
    next_state, reward_p1, done = env.step(action_p1, env.p1_marker)
    # play opponent move
    if done == False:
      state_p2 = np.array(next_state)
      action_p2 = q_network_p2.eGreedyAction(state[np.newaxis,:])
      next_state, reward_p2, done = env.step(action_p2, env.p2_marker)
    
    # check if game over
    p1win = True
    if done == True:
      if reward_p2 == 100:
        p1win = False
        reward_p1 = -100
        reward_p2 = - 100
      elif reward_p1 == 100:
        p1win = True
        reward_p1 = -100
        reward_p2 = -100
      if reward_p2 == 10 or reward_p1 == 10:
        reward_p1 = 10
        reward_p2 = 10

    q_network_p1.storeExperience(state_p1, action_p1, reward_p1, next_state, done)
    q_network_p1.updateModel()
    q_network_p2.storeExperience(state_p2, action_p2, reward_p2, next_state, done)
    q_network_p2.updateModel()
    state = np.array(next_state)
    if done == True:
      if reward_p1 == 10:
        draw += 1
      elif p1win == True:
        p1_won += 1
        p2_lost += 1
      elif p1win == False:
        p1_lost += 1
        p2_won += 1
      p1_episode_history.append(reward_p1)
      p2_episode_history.append(reward_p2)
      break

  # print status every 100 episodes
  if i_episode % 100 == 99:
    p1_mean_rewards = np.mean(p1_episode_history)
    p2_mean_rewards = np.mean(p2_episode_history)
    print("Episode {}".format(i_episode))
    print("P1 Reward for this episode: {}".format(reward_p1))
    print("P2 Reward for this episode: {}".format(reward_p2))
    print("Average P1 reward for last 100 episodes: {}".format(p1_mean_rewards))
    print("Average P2 reward for last 100 episodes: {}".format(p2_mean_rewards))
    #print("cheated:" + str(cheated))
    print("P1 lost:" + str(p1_lost))
    print("P1 won:" + str(p1_won))
    print("P2 lost:" + str(p2_lost))
    print("P2 won:" + str(p2_won))
    print("draw:" + str(draw))
    # update tensorboard
    sess.run(summary_ops[0], feed_dict = {summary_placeholders[0]:float(p1_mean_rewards)})
    result = sess.run(summaries)
    writer.add_summary(result, i_episode)

    p1_lost = 0
    p1_won = 0
    p2_lost = 0
    p2_won = 0
    draw = 0
    #cheated = 0

    # save checkpoint
    saver.save(sess, "model/saved_network")
