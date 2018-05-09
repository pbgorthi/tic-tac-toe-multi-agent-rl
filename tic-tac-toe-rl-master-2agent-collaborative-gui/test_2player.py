import tensorflow as tf
import numpy as np
from collections import deque
from rl.deep_q_network import DeepQNetwork
from game import Game
from gui import GUI
import time
import threading

class Worker(threading.Thread):
    def setProperties(self, env, gui, q_network_p1, q_network_p2):
        self.env = env
        self.gui = gui
        self.q_network_p1 = q_network_p1
        self.q_network_p2 = q_network_p2

    def run(self):
        done = False
        state = np.array(self.env.reset())
        while done == False:
            s = state[np.newaxis,:]
            print(s)
            p1_action = self.q_network_p1.eGreedyAction(s, False)
            print("P1 move: " + str(p1_action))
            next_state, p1_reward, done = self.env.step(p1_action, env.p1_marker)
            state = np.array(next_state)
            s = state[np.newaxis,:]
            self.gui.update(s[0], self.env.get_winning_combo(s[0]), done)
            time.sleep(1)
            print(s)
            if done == False:
                p2_action = self.q_network_p2.eGreedyAction(s, False)
                next_state, p2_reward, done = self.env.step(p2_action, env.p2_marker)
                print("P2 move: " + str(p2_action))
            state = np.array(next_state)
            s = state[np.newaxis,:]
            self.gui.update(s[0], self.env.get_winning_combo(s[0]), done)
            if p2_reward==10 or p1_reward==10:
                print("Draw")
            s = state[np.newaxis,:]
            print(s)
            time.sleep(1)
        print("Game over")


# initialize game env
env = Game()

# initialize tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

# define policy neural network
state_dim   = 9
num_actions = 9

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

print("Testing the learnt model")
state = np.array(env.reset())
s = state[np.newaxis,:]
gui = GUI(s[0])
w = Worker()
w.setProperties(env, gui, q_network_p1, q_network_p2)
w.start()
gui.mainloop()

