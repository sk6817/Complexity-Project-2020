'''
COMPLEXITY PROJECT

FILE: OSLO CLASS

'''
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

class Oslo:
    
    def __init__(self, L, prob = 0.5, seed = 1):
        '''
        Initialises an Oslo object
        '''
        
        self.L = L               # number of sites
        self.prob = prob         # probability for threshold slope
        self.z = np.zeros(L)     # initialise slope array
        self.h = np.zeros(L)     # initialise height array
        self.seed = seed
        np.random.seed(seed)
        self.z_th = np.random.randint(1,3,L)      # initialise threshold slope
        self.h_max = []          # list for h_max for each grain added
        self.s_list = []         # avalanche size list
        self.s = 0
        self.t_c_list = []       # cross-over time
    
    def h_max2(self):                         # final height 
      z = self.z
      return sum(z)

    def reset(self):
        
      '''
        Resets the object back to the initialised state
      '''
      self.z = np.zeros(self.L)    
      self.h = np.zeros(self.L) 
      np.random.seed(self.seed)
      self.z_th = np.random.randint(1,3,self.L)
      self.h_max = []
      self.s_list = []
      self.s = 0
      self.t_c_list = []

    def prob_th(self):
      if random.random() <= self.prob:
        return int(1)
      else:
        return int(2)

    def relax(self):
       '''
       Relaxes all sites in a system after each drive.
       '''
       z = self.z  
       z_th = self.z_th
       L = self.L
       h = self.h
       self.s = 0

       def check():
         '''
         Checks if a slope is higher than the threshold value
         '''
         check = (z - z_th > 0) 
         return check
       
       while any(check()) == True:          # do until all sites are relaxed
          
           for i in range(L):
             if z[i] > z_th[i]:
               
               if i == 0:            # first site
               
                   z[0] -= 2         # slope update
                   z[1] += 1
                   
                   h[0] -= 1         # height update
                   h[1] += 1
                  
                   self.s += 1

               elif i == L-1:        # final site
               
                   z[L-1] -= 1
                   z[L-2] += 1
                   
                   h[L-1] -= 1
                   
                   self.s += 1

                   # crossover time
                   t_c = sum(z[i-1]*i for i in range(1,L+1))
                   self.t_c_list.append(t_c) 
                                  
               else:
                   z[i] -= 2
                   z[i-1] += 1
                   z[i+1] += 1
                   
                   h[i] -= 1
                   h[i+1] +=1
                   
                   self.s += 1  

               z_th[i] = self.prob_th()

    def drive(self, N):      
        '''
        Adds a grain at site i = 1, N times.
        '''
        for i in range(N):              # add grain N times
            self.z[0] += 1
            self.h[0] += 1
            self.relax()  
                        
            self.h_max.append(self.h_max2())      # pile height after each drive
            if self.s == 0:
              self.s_list.append(0)
            if self.s != 0:
              self.s_list.append(self.s)
            
    def run(self, N, plot = False):
        '''
        Run the simulation, by driving N times.
        '''
        self.drive(N)
        h_max = self.h_max

        #figure of pile height after each addition of grain
        if plot ==  True:
          ax = plt.figure(1).add_subplot(111)    
          ax.set(title='Pile Height Distribution',
           xlabel='t (number of grains added)',
           ylabel='Pile Height')    
          plt.step(range(N), h_max, color = 'red')
          plt.show()            
    
    def h_mean(self, x = 0):               # mean height over all iterations
        h = self.h_max
        return np.mean(h[x:])

    def t_c(self):                         # cross over time
      return (self.t_c_list[0])

    def average_height(self):      # average after it has reached steady state
      t_c = self.t_c() + int(100)
      return self.h_mean(int(t_c))

    def std_h(self):                  # standard deviation
      t_c = self.t_c() + int(100)
      h_p = self.h_max[int(t_c):]
      h_sq = np.mean([x**2 for x in h_p])
      h_mean_sq = (self.average_height())**2
      std = (h_sq - (h_mean_sq))**(0.5)
      return std

    def height_prob(self):          # height probability
      t_c = self.t_c() + int(100)
      h_p = self.h_max[int(t_c):]
      T = len(h_p)
      counter=collections.Counter(h_p)    # use collections class in Python
      x = list(counter.keys())     # h
      y = list(counter.values())   # corresponding frequency of h
      prob_h = [i/T for i in y]
      x_s = [i for i,_ in sorted(zip(x,prob_h))]   # sorting in increasing order
      y_s = [_ for i,_ in sorted(zip(x,prob_h))] 
      return [x_s, y_s]
    
    def get_s(self):                       # get avalanche sizes s after t_c
      t_c = self.t_c() + int(100)
      s_after_tc = self.s_list[int(t_c):]
      return s_after_tc
  
# preparing data for L = 4, 8, 16, 32, 64, 128, 256, 512
L_list = [4, 8, 16, 32, 64, 128, 256, 512]

Oslo_piles_1 = [Oslo(4, seed = 1), Oslo(8, seed = 1), Oslo(16, seed = 1), Oslo(32, seed = 1), 
                Oslo(64, seed = 1), Oslo(128, seed = 1), Oslo(256, seed = 1), Oslo(512, seed = 1)]
Oslo_piles_2 = [Oslo(4, seed = 2), Oslo(8, seed = 2), Oslo(16, seed = 2), Oslo(32, seed = 2), 
                Oslo(64, seed = 2), Oslo(128, seed = 2), Oslo(256, seed = 2), Oslo(512, seed = 2)]
Oslo_piles_3 = [Oslo(4, seed = 3), Oslo(8, seed = 3), Oslo(16, seed = 3), Oslo(32, seed = 3), 
                Oslo(64, seed = 3), Oslo(128, seed = 3), Oslo(256, seed = 3), Oslo(512, seed = 3)]
Oslo_piles_4 = [Oslo(4, seed = 4), Oslo(8, seed = 4), Oslo(16, seed = 4), Oslo(32, seed = 4), 
                Oslo(64, seed = 4), Oslo(128, seed = 4), Oslo(256, seed = 4), Oslo(512, seed = 4)]  
Oslo_piles_5 = [Oslo(4, seed = 5), Oslo(8, seed = 5), Oslo(16, seed = 5), Oslo(32, seed = 5), 
                Oslo(64, seed = 5), Oslo(128, seed = 5), Oslo(256, seed = 5), Oslo(512, seed = 5)] 
Piles = [Oslo_piles_1, Oslo_piles_2, Oslo_piles_3, Oslo_piles_4, Oslo_piles_5]          # containing different seeds; different initial realisations      

N = 1000000      # number of iterations

#%%
# running the simulation  - takes some time
if __name__ == '__main__':      # only execute in this file, not while imported
    
    # these need to be averaged for 5 times:
    h_max = []
    t_c = []

    # for t_c and h_max averages
    for i in Piles:     
        h_max_pile = []
        t_c_pile = []
        for k in i:
            k.run(N)
            t_c_pile.append(k.t_c())
            h_max_pile.append(k.h_max)
            k.reset()
            t_c.append(t_c_pile)
            h_max.append(h_max_pile)

    h_max_avg = np.mean(h_max, axis = 0)
    t_c_avg = np.mean(t_c, axis = 0)

    # for other quantities, use the first pile with seed 1

    mean_h_all = []
    mean_h_steady = []
    std_h = []
    h_prob_h = []
    h_prob_prob = []
    s = []

    for i in Oslo_piles_1:    # for each L 
        i.run(N)
        mean_h_all.append(i.h_mean())
        mean_h_steady.append(i.average_height())
        std_h.append(i.std_h())
        h_prob_h.append(i.height_prob()[0])
        h_prob_prob.append(i.height_prob()[1])
        s.append(i.get_s())
        i.reset()

    # save simulation data as a list
    # save the list on a pickle file to load and save
    import pickle

    data = [h_max_avg , t_c_avg , mean_h_all, mean_h_steady, std_h, h_prob_h,
            h_prob_prob, s]
    with open('simulation_data.pkl', 'wb') as f:
        pickle.dump(data, f)   