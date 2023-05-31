import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def naive_hmm(p_stay, p_switch, high_reward, low_reward, N):
    
    # state = 1: cue 1 gives high reward and cue 2 gives low reward
    # state = 2: cue 2 gives low reward and cue 1 gives high reward
    
    states = np.full((N+1,), 0)
    states[0] = np.random.choice([1, 2], p = [0.5, 0.5]) # initialise randomly
    correct_cue = np.full((N,), 0)
    reward = np.full((N,2), 0.0) # rewards for choosing cue 1 or 2 on each trial

    for i in range(N): 
        
        # generating reward and next state based on current state 
        if states[i] == 1:
            
            correct_cue[i] = 1 # correct cue is 1 which gives higher reward
            reward[i,0] = np.random.choice(high_reward) # high reward for cue 1
            reward[i,1] = np.random.choice(low_reward) # low reward for cue 2
            states[i+1] = np.random.choice([1,2], p = [p_stay, p_switch])
            
        elif states[i] == 2:
            
            correct_cue[i] = 2 # correct cue is 2 which gives higher reward
            reward[i,0] = np.random.choice(low_reward) # low reward for cue 1
            reward[i,1] = np.random.choice(high_reward) # high reward for cue 2
            states[i+1] = np.random.choice([1,2], p = [p_switch, p_stay])

    return correct_cue, reward

def block_hmm(p_stay, p_switch, high_reward, low_reward, N, block_size):
    
    # state = 1: cue 1 gives high reward and cue 2 gives low reward
    # state = 2: cue 2 gives low reward and cue 1 gives high reward
    
    state = np.random.choice([1, 2], p = [0.5, 0.5]) # initialise randomly
    correct_cue = np.full((N+block_size,), 0)
    reward = np.full((N+block_size,2), 0.0) # rewards for choosing cue 1 or 2 on each trial
    
    i = 0
    
    while i < N : 
        
        # generating reward and next state based on current state 
        if state == 1:
            
            correct_cue[i:i+block_size] = 1 # correct cue is 1 which gives higher reward
            reward[i:i+block_size,0] = np.random.choice(high_reward, size=block_size) # high reward for cue 1
            reward[i:i+block_size,1] = np.random.choice(low_reward, size=block_size) # low reward for cue 2
            state = np.random.choice([1,2], p = [p_stay, p_switch])
            
        elif state == 2:
            
            correct_cue[i:i+block_size] = 2 # correct cue is 2 which gives higher reward
            reward[i:i+block_size,0] = np.random.choice(low_reward, size=block_size) # low reward for cue 1
            reward[i:i+block_size,1] = np.random.choice(high_reward, size=block_size) # high reward for cue 2
            state = np.random.choice([1,2], p = [p_switch, p_stay])
        
        i = i+block_size
        
    return correct_cue, reward
    
#%%
    
p_stay = 0.98
p_switch = 0.02
high_reward = np.arange(4, 9.1, 0.1)
low_reward = np.arange(0, 5.1, 0.1)
N = 200 # number of trials
correct_cue, reward = naive_hmm(p_stay, p_switch, high_reward, low_reward, N)
        
trials = np.arange(1, N+1, 1)
plt.plot(trials, correct_cue[:N])
plt.xlabel('trials')
plt.ylabel('state')
plt.figure()
plt.plot(trials, reward[:N,0], label = 'reward for cue 1')
plt.plot(trials, reward[:N,1], label = 'reward for cue 2')
plt.xlabel('trials')
plt.ylabel('reward')
plt.legend()  

#%%
    
p_stay = 0.4
p_switch = 0.6
high_reward = np.arange(4, 9.1, 0.1)
low_reward = np.arange(0, 5.1, 0.1)
N = 200 # number of trials
block_size = 20
correct_cue, reward = block_hmm(p_stay, p_switch, high_reward, low_reward, N, block_size)
        
trials = np.arange(1, N+1, 1)
plt.plot(trials, correct_cue[:N])
plt.xlabel('trials')
plt.ylabel('state')
plt.figure()
plt.plot(trials, reward[:N,0], label = 'reward for cue 1')
plt.plot(trials, reward[:N,1], label = 'reward for cue 2')
plt.xlabel('trials')
plt.ylabel('reward')
plt.legend() 

# generating csv file for experiment

# correct key response is left for cue=1 and right for cue=2
correct_keys = np.where(correct_cue[:N] == 1, "[\'left\']", "[\'right\']").reshape(N,1) 
correct_reward = np.where(correct_cue[:N] == 1, reward[:N,0], reward[:N,1]).reshape(N,1) # reward on corrcet ans
wrong_reward = np.where(correct_cue[:N] == 1, reward[:N,1], reward[:N,0]).reshape(N,1) # reward on wrong ans
reward_data = pd.DataFrame(np.hstack((correct_keys, np.round(correct_reward,2), np.round(wrong_reward,2))), 
              columns = ['correctAns', 'correct_reward', 'wrong_reward'], dtype = object)

reward_data.to_csv('reward.csv', index=False)

#%%
#1000 experiments
p_stay = 0.4
p_switch = 0.6
high_reward = np.arange(4, 9.1, 0.1)
low_reward = np.arange(0, 5.1, 0.1)
N = 200 # number of trials
block_size = 20
cue_counts = np.full((1000,2), 0) # counts of corrcet cues in each experiment
for i in range(1000):
    correct_cue, reward = block_hmm(p_stay, p_switch, high_reward, low_reward, N, block_size)
    cue_counts[i,0] = len(np.where(correct_cue[:N]==1)[0])
    cue_counts[i,1] = len(np.where(correct_cue[:N]==2)[0])
    
plt.boxplot(cue_counts)