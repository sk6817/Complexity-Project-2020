"""
COMPLEXITY PROJECT

FILE: TASK 1
"""
from Oslo_class import Oslo
import numpy as np
import matplotlib.pyplot as plt


# parameters for plotting

params = {
       'axes.labelsize': 15,
       'font.size':15,
       'legend.fontsize': 12,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'figure.figsize': [6, 4],
       'figure.dpi': 100,
       'axes.grid': 1,
        }
plt.rcParams.update(params)

# Testing the model
# p = 1 is BTW model 

ax1 = plt.figure(1).add_subplot(111)    
ax1.set(#title='Pile Height Distribution',
           xlabel='t (number of grains added)',
           ylabel='Pile Height')    

pile_16_btw = Oslo(16, prob = 1)                 # pile of 16 sites with p = 1
pile_16_btw.run(500, plot = 0)
pile_16_btw_h_max_list = pile_16_btw.h_max
print('The final height is',pile_16_btw.h)
print('The final slope is',pile_16_btw.z)
# p = 0

pile_8_0 = Oslo(16, prob = 0)                 # pile of 8 sites with p = 0
pile_8_0.run(500, plot = 0)
pile_16_btw_h_max_list_2 = pile_8_0.h_max
print('The final height is',pile_8_0.h)
print('The final slope is',pile_8_0.z)

plt.step(range(500), pile_16_btw_h_max_list, color = 'red', label = 'p = 1')
plt.step(range(500), pile_16_btw_h_max_list_2, color = 'blue', label = 'p = 0')
plt.legend(loc='lower right')
plt.show()  

# testing the Oslo model (p = 0.5)

pile_16_test = Oslo(16, prob = 0.5)              # pile of 16 sites with p = 0.5
pile_16_test.run(500, plot = 0)
pile_16_test_h_max_list = pile_16_test.h_max

# plotting BTW and Oslo together for L = 16

ax = plt.figure(2).add_subplot(111)    
ax.set(#title='Pile Height Distribution',
           xlabel='t (number of grains added)',
           ylabel='Pile Height')    
plt.step(range(500), pile_16_btw_h_max_list, color = 'red', label = 'BTW with L = 16')
plt.step(range(500), pile_16_test_h_max_list, color = 'blue', label = 'Oslo with L = 16')
plt.legend(loc='lower right')
plt.show()      

#%%

# checking mean height for L = 16

pile_16_test_2 = Oslo(16, prob = 0.5, seed = 1)              # pile of 16 sites with p = 0.5
pile_16_test_2.run(10000)
print('The mean height for L = 16 is', pile_16_test_2.h_mean(300))

# checking mean height for L = 32

pile_32_test = Oslo(32, prob = 0.5, seed = 1)                # pile of 32 sites with p = 0.5
pile_32_test.run(10000)
pile_32_test_h_max_list = pile_32_test.h_max
print('The mean height for L = 32 is', pile_32_test.h_mean(1000))

#%%
# own test for oslo model
# comparing with analytical resuls for number of recurrent states
import random

l_2 = 2       # system size

z_len_2 = []
for j in range(10):
  z_set_2 = []
  for i in range(10000):
    oslo_2_2 = Oslo(l_2, seed = random.randint(1, 2000000))
    oslo_2_2.run(10)
    z_set_2.append(oslo_2_2.z)
    oslo_2_2.reset()
  z_uniq_2 = np.unique(z_set_2, axis=0)    # must be unique states
  z_len_2.append(len(z_uniq_2))

n_r_2 = np.mean(z_len_2)         # recurrent config.
prob_n_r_2 = n_r_2/(3**l_2)
print('Number of recurrent states for L = 2 is', n_r_2, '+-', np.std(z_len_2))
print('Probability of a recurrent state for L = 2 is', prob_n_r_2)


l = 4        # system size

z_len = []
for j in range(20):
  z_set = []
  for i in range(100000):
    oslo_2 = Oslo(l, seed = random.randint(1, 2000000))
    oslo_2.run(25)
    z_set.append(oslo_2.z)
    oslo_2.reset()
  z_uniq = np.unique(z_set, axis=0)
  z_len.append(len(z_uniq))

n_r = np.mean(z_len)         # recurrent config.
prob_n_r = n_r/(3**l)
print('Number of recurrent states for L = 4 is', n_r, '+-', np.std(z_len))
print('Probability of a recurrent state for L = 4 is', prob_n_r)    

#Number of recurrent states for L = 2 is 5.0 +- 0.0
#Probability of a recurrent state for L = 2 is 0.5555555555555556
#Number of recurrent states for L = 4 is 34.0 +- 0.0
#Probability of a recurrent state for L = 4 is 0.41975308641975306