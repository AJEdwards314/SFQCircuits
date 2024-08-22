import numpy as np
import matplotlib.pyplot as plt 

# Code designed to create and insert specified SFQ pulse parameters into test file

#TODO enable method to insert inputs 

amplitude = 900e-6                    # Setting basic parameters for impulse height 
constant = 2.06783e-15
width = ((constant) * 2) / amplitude
delay = 10e-12
freq = 100000000000                     # Frequency at which pulses occur 
period = 1/freq
multi = 1                                 
unit = ''

num_spikes = 1                          # Number of pulses which will occur


if freq / 1000 >= 1:                    # Determine the input frequency with corresponing unit 
    unit = 'm'
    multi = 1000
if freq / 1000000 >= 1:
    unit = 'u'
    multi = 1000000
if freq / 1000000000 >= 1:
    unit = 'n'
    multi = 1000000000
if freq / 1000000000 >= 1:
    unit = 'p'
    multi = 1000000000

xval = []                               # Initialize lists for integer and string 
strXval = []
yval = []
strYval = []
pwl = []

for i in range(num_spikes):             # Populate X values 
    if i == 0:                          # Initial point must be 0
        yval.append(0)
    if i >= 0: 
        yval.append(0)
        yval.append(amplitude)
        yval.append(0)
        

count = 1
for i in range(len(yval)):              # Populate X values
    if i == 0:                          # Initial point must be 0 
        x = 0
        xval.append(x)

    if yval[i] == amplitude and count == 1:             # Includes delay before train of impulses 
        x = delay
        xval.insert(i - 1, x)
        xval.insert(i, x + width/2)
        xval.insert(i + 1, x + width)
        count += 1 
    elif yval[i] == amplitude and count > 1:            # Includes all spikes after first 
        x += period
        xval.insert(i - 1, x)
        xval.insert(i, x + width/2)
        xval.insert(i + 1, x + width)
        count += 1 

'''
Sample Signal from Basic JTL example 
xp = [0, 300e-12, 302.5e-12, 305e-12, 600e-12, 602.5e-12, 605e-12] 
yp = [0, 0,       827.13e-6, 0,       0,       827.13e-6, 0]
plt.plot(xp, yp)
plt.show()
'''
          
for i in range(len(yval)): 
    if i == 0: 
        pwl.append(str(round(xval[i]*1000000000000, 4)))
        pwl.append(str(yval[i]*1000000)) 
    elif i != 0: 
        pwl.append(str(round(xval[i]*1000000000000, 4)) +'p')
        pwl.append(str(yval[i]*1000000) + 'u') 

Spwl = ', '.join(pwl)

print(Spwl)

plt.plot(xval, yval)
plt.show()