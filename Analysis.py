'''
COMPLEXITY PROJECT

FILE: TASK 2 & 3 

'''
# load libraries and modules from other files
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from Functions import power_law
from Functions import linear
from Functions import scaling_2e
from Functions import logbin

# load the simulation data saved as a pickle file

with open('simulation_data.pkl', 'rb') as f:
    data = pickle.load(f)

h_max_avg = data[0] 
t_c_avg = data[1] 
mean_h_all = data[2] 
mean_h_steady = data[3] 
std_h = data[4] 
h_prob_h = data[5] 
h_prob_prob = data[6] 
s = data[7]

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

N = 1000000         # number of grains added
L_list = [4, 8, 16, 32, 64, 128, 256, 512]   # list of L's iused

#%%
# task 2a - height distribution

# predict how h varies with t during transient (for task 2d)
cr = 50000
x_2d_max = np.arange(1,len(h_max_avg[-1])+1)

# fitting the power law
(A_1, B_1), cov_2d = curve_fit(power_law, x_2d_max[:cr], h_max_avg[-1][:cr])
error_2d = np.sqrt(np.diag(cov_2d))

# log plot 
ax2a2 = plt.figure(2).add_subplot(111)    
ax2a2.set(#title='Pile Height Distribution',
           xlabel='$t$ (number of grains added)',
           ylabel='Pile Height') 
for j in range(len(h_max_avg)):
  plt.loglog(range(1, N+1), h_max_avg[j], '.', ms = 1.5, label = 'L = %r' %L_list[j])
  
#plt.loglog(x_2d_max, power_law(x_2d_max, A_1, B_1))
plt.legend(markerscale = 4, fontsize ='small')
plt.show()

print(B_1, '+-', error_2d[1])  # the exponent at transient state
print(A_1,'+-', error_2d[0])   # the coefficient of power law

#%%
# task 2b - cross-over time

# fitting the power law
(A, B), cov_2b = curve_fit(power_law, L_list[-4:], t_c_avg[-4:])        
error_2b = np.sqrt(np.diag(cov_2b))
ax3 = plt.figure(3).add_subplot(111)    
ax3.set(#title='Pile Height Distribution',
           xlabel='$L$',
           ylabel=r"${\langle}t_c{\rangle}$")    
plt.loglog(L_list, t_c_avg, '.', label = 'data')                      
plt.loglog(L_list, power_law(L_list, A, B), 'r', label = 'fit')
plt.legend(fontsize ='small')
plt.show()       

print('The exponent of <t_c> over L is', B, '+-', error_2b[1])      # the exponent ~ 2
print(A, '+-', error_2b[0])    # the coefficient of power law above
#%%
# task 2d
# data collapse

ax2d = plt.figure(4).add_subplot(111)     
ax2d.set(#title='Pile Height Distribution',
           xlabel='$t/{L^2}$',
           ylabel='$h/L$')   

for i in range(len(L_list)):
  x_2d = np.arange(1,len(h_max_avg[i])+1)/(L_list[i]**2)
  r = L_list[i]
  y_2d = [k/r for k in h_max_avg[i]]
  plt.loglog(x_2d, y_2d, '.', ms =2, label = 'L = %r' %L_list[i])

plt.legend(markerscale = 3)
plt.show()      

# plotting zoomed-in version
ax2d_2 = plt.figure(5).add_subplot(111)     
ax2d_2.set(#title='Pile Height Distribution',
           xlabel='$t/{L^2}$',
           ylabel='$h/L$')   

for i in range(len(L_list)):
  x_2d = np.arange(1,len(h_max_avg[i])+1)/(L_list[i]**2)
  r = L_list[i]
  y_2d = [k/r for k in h_max_avg[i]]
  plt.loglog(x_2d, y_2d, '.', ms = 3, label = 'L = %r' %L_list[i])

plt.legend(markerscale = 2)
plt.xlim(0.585, 1.516)
plt.ylim(1.36, 1.89)
plt.show()      

#%%
# task 2e - average height

# fitting the power law
(a_0, a_1, w_1), cov_2e = curve_fit(scaling_2e, L_list, mean_h_steady)
print(a_0, a_1, w_1)
error_2e = np.sqrt(np.diag(cov_2e))
print(error_2e)

x_2e = np.linspace(1, max(L_list), 200)
ax2e = plt.figure(6).add_subplot(111)    
ax2e.set( xlabel='$L$',
           ylabel= r"${\langle}h{\rangle}$")    
plt.plot(L_list, mean_h_steady, '.', label = 'data')
plt.plot(x_2e, scaling_2e(x_2e, a_0, a_1, w_1), 'r', label = 'fit')
plt.legend()
plt.show()       

# plotting the scaled height
h_scaled = np.zeros(len(L_list))
for i in range(len(L_list)):
  h_scaled [i] = mean_h_steady[i] / (a_0 * L_list[i])

ax2e2 = plt.figure(7).add_subplot(111)    
ax2e2.set( xlabel='$L$',
           ylabel= r"${\langle}h{\rangle}/{a_0 L}$")    
plt.plot(L_list, h_scaled, '.', label = 'data')
plt.plot(x_2e, scaling_2e(x_2e, a_0, a_1, w_1)/ (a_0*x_2e), 'r', label = 'fit')
plt.legend()
plt.show()       

#%%

# task 2f - standard deviation
x_2e = np.linspace(1, max(L_list))
(C, D), cov_2f = curve_fit(power_law, L_list, std_h)
print(C, D)
error_2f = np.sqrt(np.diag(cov_2f))
print(error_2f)

ax2f = plt.figure(8).add_subplot(111)    
ax2f.set(   xlabel='$L$',
           ylabel=r"$\sigma_h$")    
plt.loglog(L_list, std_h, '.', label = 'data')
plt.loglog(L_list, power_law(L_list, C, D), color = 'r', label = 'fit' )
plt.legend()
plt.show()       

# checking for scaling effect
std_scaled = np.zeros(len(L_list))
for i in range(len(L_list)):
  std_scaled [i] = std_h[i] / (C * (L_list[i] ** D))

ax2f2 = plt.figure(9).add_subplot(111)    
ax2f2.set(   xlabel='$L$',
           ylabel=r"$\sigma_h / AL^{\alpha}$")    
plt.plot(L_list, std_scaled, '.', label = 'data')
plt.plot(x_2e, power_law(x_2e, C, D)/ (C * (x_2e ** D)), 'r')
plt.ylim(0.95, 1.05)
plt.legend()
plt.show()       

#%%
# task 2g  - probability of height

ax2g = plt.figure(10).add_subplot(111)    
ax2g.set(#title='Height Probability',
           xlabel='$h$',
           ylabel='$P(h;L)$')  
y_max_2g = []
for i in range(len(L_list)):
  x_2g = h_prob_h[i]
  y_2g = h_prob_prob[i]
  y_max_2g.append(max(y_2g))
  f =interp1d(x_2g, y_2g)
  plt.plot(x_2g, f(x_2g), label = 'L = %r' %L_list[i], linewidth = 1.2)

plt.plot(mean_h_steady, y_max_2g, '.', color = 'black')
plt.ylim(0, 0.5)
plt.legend(markerscale = 4)  
plt.show() 

# data collapse on height probability
ax2g2 = plt.figure(11).add_subplot(111)    
ax2g2.set(#title='Height Probability',
           xlabel= r'$(h - {\langle}h{\rangle})/\sigma_h$',
           ylabel='${\sigma_h}P(h;L)$ ')  

for i in range(len(L_list)):
  x_2g = h_prob_h[i]
  y_2g = h_prob_prob[i]
  h_mean_2g = mean_h_steady[i]
  std_L_2g = std_h[i]
  x_2g_1 = x_2g - h_mean_2g
  x_2g_2 = x_2g_1 / std_L_2g
  y_2g_1 = [j*std_L_2g for j in y_2g]
  plt.plot(x_2g_2, y_2g_1, '.', ms = 4, label = 'L = %r' %L_list[i], linewidth = 1.2)

plt.legend(markerscale = 2)  
plt.show() 

#%%
# task 3a   - avalanche size probability plot

ax3a = plt.figure(12).add_subplot(111)    
ax3a.set(#title='Avalanche-Size Probability',
           xlabel="$s$",
           ylabel=r"$P_N(s;L)$")  

for i in range(len(L_list)):
  x_bin_3a, y_bin_3a = logbin(s[i], scale= 1.15, zeros = 1)
  plt.loglog(x_bin_3a[1:], y_bin_3a[1:], '.', ms = 4,label = 'L = %r' %L_list[i], linewidth = 1.2)

plt.legend(markerscale = 2, fontsize ='small') 
plt.show()

#%%
# task 3b  - data collapse on avalanche size probability

# power law on the linear part
# use largest L
s_64 = s[-1]
x_bin_64, y_bin_64 = logbin(s_64, scale= 1.15, zeros = 1)
x_lin = x_bin_64[20:-40]
y_lin = y_bin_64[20:-40]

(G, tau_s), p3cov = curve_fit(power_law, x_lin, y_lin)
print(G, tau_s)
error_3b = np.sqrt(np.diag(p3cov))
print(error_3b)

# plot of fit
x_range = np.linspace(1, 10*N, 100)
ax3b_1 = plt.figure(13).add_subplot(111)    
ax3b_1.set(#title='Avalanche-Size Probability',
           xlabel="$s$",
           ylabel=r"$P_N(s;L)$")  
plt.loglog(x_bin_64, y_bin_64, '.', label = 'data')
plt.loglog(x_range, power_law(x_range, G, tau_s), '-', label = 'fit')
plt.legend()
plt.show()

#first collapse - horizontal

ax3b = plt.figure(14).add_subplot(111)    
ax3b.set(#title='Avalanche-Size Probability',
           xlabel="$s$",
           ylabel=r"$s^{\tau_s}P_N(s;L)$")  

for i in range(len(L_list)):
  x_bin_3b, y_bin_3b = logbin(s[i], scale= 1.15, zeros = 1)    
  x_bin_sc = [i**(-tau_s) for i in x_bin_3b]
  y_bin_3b = np.multiply(y_bin_3b, x_bin_sc)
  plt.loglog(x_bin_3b[1:], y_bin_3b[1:], '.', ms = 4, label = 'L = %r' %L_list[i], linewidth = 1.2)

plt.legend(markerscale = 2) 
plt.show()

# full data collapse

Exp_D = 2.25    # try this (obtained from 3c - moment analysis)

ax3b2 = plt.figure(15).add_subplot(111)    
ax3b2.set(#title='Avalanche-Size Probability',
           xlabel="$s/L^D$",
           ylabel=r"$s^{\tau_s}P_N(s;L)$")  

for i in range(0, len(L_list)):
  x_bin_3b2, y_bin_3b2 = logbin(s[i], scale= 1.15, zeros = 1)   
  x_bin_sc_2 = [i**(-tau_s) for i in x_bin_3b2]
  y_bin_3b2 = np.multiply(y_bin_3b2, x_bin_sc_2 )
  x_bin_3b2 = [j/(L_list[i]**Exp_D) for j in x_bin_3b2]
  plt.loglog(x_bin_3b2[1:], y_bin_3b2[1:], '.', ms = 4, label = 'L = %r' %L_list[i], linewidth = 1.2)

plt.legend(loc = 'lower left', markerscale = 2) 
plt.show()

#%%
# task 3c -  moment analysis

k_list = [1,2,3,4]
def k_moment(k, index):
    # calculates k'th moments for a given s list
  return np.mean([i**k for i in s[index]])
s_k_L = []

for j in range(1,len(k_list) +1):    # groups the moments together
  s_L_k = []
  for i in range(len(L_list)):
    s_L_k.append(k_moment(j, i))
  s_k_L.append(s_L_k)

# plotting the k moments
ax3c = plt.figure(16).add_subplot(111)    
ax3c.set(#title='k-moment of s against L',
           xlabel="$L$",
           ylabel=r"${\langle}s^k{\rangle}$") 

x_3c = np.linspace(1, L_list[-1], 100)
power = []             # the exponent list for each k
power_error = []
for i in range(len(s_k_L)):
  plt.loglog(L_list, s_k_L[i], '.', color = 'black') 
  (alpha, beta), cov_3c = curve_fit(linear, np.log(L_list[-4:]), np.log(s_k_L[i][-4:]))                         
  power.append(alpha)
  error_alpha = np.sqrt(np.diag(cov_3c))[0]
  power_error.append(error_alpha)
  plt.loglog(x_3c, power_law(x_3c, np.exp(beta), alpha), label = 'k = %r' %k_list[i])

plt.legend(loc = 'upper left') 
plt.show()

# fitting the exponent plot
(m, c), cov_lin = curve_fit(linear, k_list, power)
error_d = np.sqrt(np.diag(cov_lin))
error_tau = np.sqrt((c*error_d[0]/(m**2))**2 + (error_d[1]/m)**2)
print('D is', m, '+-', error_d[0])
print('Tau is', 1 - c/m, '+-', error_tau)

ax3c2 = plt.figure(17).add_subplot(111)    
ax3c2.set(#title='k-moment of s against L',
           xlabel="$k$",
           ylabel=r"$D(1 + k - \tau_s)$") 
x_k = k_list
y_k = [i*m +c for i in x_k]
plt.plot(x_k, power, '.', color = 'b')
plt.plot(x_k, y_k, 'red', linewidth = 1.5)
plt.show()

print('for consistency', m*(2-(1 - c/m)), 'should be close to 1')

#corrections to scaling
k_scale = 1
E = m*(1+ k_scale - (1 - c/m))   # the exponent

def scaling_3c(L, a_0, a_1, w_1):
    '''
    Scaling function for task 3c   
    '''
    f = a_0*(L**E)*(1-a_1*L**(- w_1))
    return f

# fitting the scaling function 3c
(a_0_3c, a_1_3c, w_1_3c), cov_3c_2 = curve_fit(scaling_3c, L_list, s_k_L[k_scale-1])
print(a_0_3c, a_1_3c, w_1_3c)
error_3c_2 = np.sqrt(np.diag(cov_3c_2))
print(error_3c_2)

s_scaled = np.zeros(len(L_list))
for i in range(len(L_list)):
  s_scaled [i] = s_k_L[k_scale-1][i] / (a_0_3c * (L_list[i])**E)

ax2e3 = plt.figure(7).add_subplot(111)    
ax2e3.set( xlabel='$L$',
           ylabel= r"${\langle}s{\rangle}/{b_0 L^{D(2 - \tau_s)}}$")    
plt.plot(L_list, s_scaled, '.', label = 'data')
plt.plot(x_2e, scaling_3c(x_2e, a_0_3c, a_1_3c, w_1_3c)/ (a_0_3c*(x_2e**E)), 'r', label = 'fit')
plt.legend()
plt.show()       