"""
experiment 4
Policy return method with actor-critic.

"""

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 2000
EP_LEN = 300
GAMMA = 0.93
A_LR = 0.0005
C_LR = 0.0008
TAU = 0.98
BATCH = 32
er_bath_num = 100
er_max = -2300
RESTORE_COUNT = 200
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1


METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]       


class Model(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic',reuse=tf.AUTO_REUSE):
            self.v_t = self._build_cnet('critic_traget')
            self.v = self._build_cnet('critic_online')
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
            self.c_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/critic_traget')
            self.c_online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/critic_online')
        
        # actorA_LR
        with tf.variable_scope('actor',reuse=tf.AUTO_REUSE):
            _,_= self._build_anet('pi_t',trainable=False)
            self.t_pi_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/pi_t')
        
        pi, self.pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action',reuse=tf.AUTO_REUSE):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi',reuse=tf.AUTO_REUSE):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('surrogate',reuse=tf.AUTO_REUSE):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain',reuse=tf.AUTO_REUSE):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        with tf.variable_scope('update_target'):
            self.target_replace = [[ta.assign(ea), tc.assign(ec)]
                             for ta, ea, tc, ec in zip(self.t_pi_params, self.pi_params, self.c_target_params, self.c_online_params)]
        
        with tf.variable_scope('online_go_back'):
            self.online_back = [[ea.assign(ta*random.uniform(1-TAU, 1+TAU)), ec.assign(tc*random.uniform(1-TAU, 1+TAU))]
                             for ta, ea, tc, ec in zip(self.t_pi_params, self.pi_params, self.c_target_params, self.c_online_params)]
        
        with tf.variable_scope('online_go_back_s'):
            self.online_back_s = [[ea.assign(ta*TAU+ea*(1-TAU)), ec.assign(tc*TAU+ec*(1-TAU))]
                              for ta, ea, tc, ec in zip(self.t_pi_params, self.pi_params, self.c_target_params, self.c_online_params)]


        # actorA_LR
        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(self.tfs,256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 64, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l2, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params
    
    def _build_cnet(self, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(self.tfs, 500, tf.nn.relu)
            v_ = tf.layers.dense(l1, 1)
        return v_
        
    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    
    def save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=True)

    def restore(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess,path)
    
    def update_target(self):
        self.sess.run(self.target_replace)
        print('update_target********')
       
        
    def soft_go_back(self):
        Mode = 0
        print('restore*******')
        if Mode == 0:
            a_list=tf.global_variables(scope = 'pi')
            c_list=tf.global_variables(scope = 'critic/critic_online')
            a_ini = tf.variables_initializer(a_list)
            c_ini = tf.variables_initializer(c_list)
            self.sess.run(a_ini)
            self.sess.run(c_ini)
            self.sess.run(self.online_back_s)
        else:
            self.sess.run(self.online_back)
        
        

def check_h(er_old,er_new,n_r):
    if abs(er_new-er_old) < 15 and er_new<n_r:
        return True
    else:
        return False

def check_h_s(n_r):
    ner =0
    for e in er_bath:
        ner += e
    ner = ner/len(er_bath)
    if ner>er_max and ner<n_r:
        return True , ner
    else:
        return False , ner
        
        
def get_er_o_n(er_bath):
    oer_s=0
    ner_s=0
    for i in range(len(er_bath)):
        if i < len(er_bath)/2:
            oer_s += er_bath[i]
        else:
            ner_s += er_bath[i]
    oer = oer_s/(len(er_bath)/2)
    ner = ner_s/(len(er_bath)/2)
    return oer,ner


def chec_restore(ep,n_er,ppo):
    global low_count,A_LR,C_LR,ReLoad,TAU
    if n_er < er_max:
        low_count+=1
    else:
        low_count = 0
    if low_count > RESTORE_COUNT and MEM_EN:
        low_count = 0
        ppo.soft_go_back()

        seed=np.random.randint(0,9)
        env.seed(seed)
        ReLoad += 1
        reload_index.append(ep)

def test(name):
    for i in range(5):
        rr=[]
        for j in range(100):
            rt=0
            s = env.reset()
            for t in range(200):      
                # env.render()
                s, r, done, _ = env.step(ppo.choose_action(s))
                rt += r
            rr.append(rt)
        dr=0
        for d in rr:
            dr+=d
        print(name,dr/len(rr))
                              
##########################################################
env = gym.make('Pendulum-v0').unwrapped
tf.reset_default_graph()
ppo = Model()
all_ep_r = []
er_bath = []

er_h_list = []
er_h_list.append([0,-2000])
res = False
er_max_list=[]
reload_index=[]
low_count = 0
DATA_PATH = ''
MEM_EN = False
ReLoad = 0
for ep in range(EP_MAX):
    er_max_list.append(er_max)
    s = env.reset()
    s_y = s
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    n_r = 0
    for t in range(EP_LEN):   # in one episode
        # if ep>(EP_MAX-1): env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        n_r = r
        # print(ep,t,s,r)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.85 + ep_r*0.15) 
    
    er_bath.append(ep_r)
    if len(er_bath)>er_bath_num:
        er_bath.remove(er_bath[0])
        res , n_er = check_h_s(n_r)
        chec_restore(ep,n_er,ppo)  
        if res and (ep-er_h_list[-1][0]) > 2 :
            er_h_list.append([ep,n_er])
            er_max = n_er
            ppo.update_target()
            MEM_EN = True
            if er_max > -250:
                DATA_PATH = './log_lr/' +str(ep)+'_'+str(int(er_max))
                ppo.save(DATA_PATH)
                
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        '|',int(er_max),    
        "reload",ReLoad      
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r,color='#00E5EE',linewidth=1)
plt.xlabel("Training Episodes");plt.ylabel('Averaged Reward');plt.show()

mypath='./data.txt'
f1=open(mypath,'w',encoding='utf-8')
for d in all_ep_r:
    f1.write(str(format(d,'.2f'))+'\n')
f1.close()
