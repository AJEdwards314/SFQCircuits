import numpy as np
import matplotlib.pyplot as plt 

# Code designed to create and insert specified SFQ pulse parameters into test file

amplitude = 827.13e-6
constant = 2.06783e-15
width = ((constant) * 2) / amplitude
delay = 300e-12
freq = 1000000000000
period = 1/freq

duration = 7
multi = 1
unit = ''
num_spikes = 2


if freq / 1000 >= 1:
    unit = 'm'
    multi = 1000
if freq / 1000000 >= 1:
    unit = 'u'
    multi = 1000000
if freq / 1000000000 >= 1:
    unit = 'n'
    multi = 1000000000
if freq / 1000000000000 >= 1:
    unit = 'p'
    multi = 1000000000000

xval = []
strXval = []
yval = []
strYval = []
pwl = []

# Sample PWL signal
# (0 0 20p 1m 40p 0 60p 1m)

for i in range(duration): 
    if (i + 1) % 3 == 0: 
        yval.append(amplitude)
    else: 
        yval.append(0)

count = 1

for i in range(len(yval)): 
    if i == 0:
        x = 0
        xval.append(x)
    for k in range(num_spikes - 1): 
        if yval[i] == amplitude: 
            x = delay * count
            xval.insert(i - 1, x)
            xval.insert(i, x + width/2)
            xval.insert(i + 1, x + width/2)
            count += 1 
        

'''
Sample Signal from Basic JTL example 

xp = [0, 300e-12, 302.5e-12, 305e-12, 600e-12, 602.5e-12, 605e-12] 
yp = [0, 0,       827.13e-6, 0,       0,       827.13e-6, 0]
plt.plot(xp, yp)
plt.show()
'''
for i in range(len(xval)): 
    strXval.append(str(round(xval[i],2)) + unit)


print(strXval)

#plt.plot(xval, yval)
#plt.show()
