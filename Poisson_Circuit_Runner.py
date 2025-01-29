# Code designed to detect peaks over some frequency sweep 
# Generates some poisson distributed variable in order to import 
# into JOSIM, where the input is swept across a range of values 

import pandas
import os 
import matplotlib.pyplot as plt
import numpy as np

# Function for creating poisson spikes 
def poisson(freq):
        
    ############################################################
    # Create poisson spikes and store the dt time interval
    sim_length = 650e-12 
    sum = 0 
    delta = 5e-12
    const = 2.0678e-15
    height = (const * 2) / (2 * delta)
    time = []
    x = []
    vin = []

    def generate_poisson_spikes(rate, gen):
        # Generate Poisson-distributed spikes
        spike = gen.exponential(1/rate)
        return spike

    rate = freq * 1e9  # Average rate (lambda) of the Poisson distribution
    gen = np.random.default_rng(0)

    while(sum <= sim_length): 
        sum += generate_poisson_spikes(rate,gen)
        time.append(sum)

    for i in range(len(time) + 1): 
        if i == 0: 
            x.append(0)
            vin.append(0)
        else:
            x.append(time[i - 1] - delta)
            vin.append(0)
            x.append(time[i - 1])
            vin.append(height)
            x.append(time[i - 1] + delta)
            vin.append(0)

    pwl = ""

    for i in range(len(x)): 
        pwl += str((x[i])) + " " + str(round(vin[i], 5)) + " "

    return pwl

    # The string PWL is completed above, and can be imported to the circuit in order to run. 
    #########################################################################################

# Function used for editing circuit file and importing the intended period for that run
def edit_circuit(pwl): 

    with open("Neuron_Alex.cir") as fh: 
        str = fh.read()

        str = str.format(pwl)

        with open("SFQ_Neuron_TESTING.cir", 'w') as fh: 
            fh.write(str)

# Lists which will contain the plotting points for phase and voltage calculated 
output_freq = []
output_phase = []

# Define points and increment value for frequency sweep 
input_freq = [] 
increment = 10
max_freq = 200 
min_freq = 5 
step = int((max_freq - min_freq) / increment)
x = 0 

# Create list for such variables 
for i in range(step + 1):
    x = min_freq + ((i)*increment)
    input_freq.append(x)

# This is where the iteration should begin, first importing the intended frequency to the circuit
# Poisson code will create pwl to import string as the argument 
# Next the os will run JOSIM and simulate under those specifications 
# Finally use the Flux_Detect in order to measure our output 

# Code is intended to send frequency to poisson function, and output the corresponding pwl 
for j in range(len(input_freq)): 

    # Create spike pwl and input to editting function 
    pwl = poisson(input_freq[j])

    # Make changes to file before running
    edit_circuit(pwl) 

    # Call JOSIM and exectute statement
    def run_josim(): 
        os.system('josim-cli -o ./SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv SFQ_Neuron_TESTING.cir')

    if __name__ == "__main__": 
        run_josim()

    # Store CSV within py 
    data = pandas.read_csv('SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv')

    # Store time, phase, and voltage as lists 
    voltage = data['V(4,0)'].tolist()
    phase = data['P(B02)'].tolist()
    time = data['time'].tolist()

    # Constant 
    const = 2.0678e-15 
    sim_length = (time[len(time) - 1]) - (time[0]) 

    # Define current, past, and arithmetic variables 
    cVAL = 0
    fVAL = 0 
    x = 0
    sum = 0 
    frequency = 0
    y = 0

    # Find the number of spikes based on phase
    iPHASE = abs(phase[0])
    fPHASE = abs(phase[-1])
    y = (fPHASE - iPHASE) / (2 * 3.14159)
    y /= sim_length
    y /= 1e9

    # Add values to phase list 
    if y > 0: 
        output_phase.insert(j, y)
    else: 
        output_phase.insert(j, 0)

    # List to be populated for DELTA X values  
    list_of_calculations = [] 
    # Parse list and populate list_of_calculations with DELTA X values multiplied by the f(x) value 
    for i in range(1, len(time)): 
        pVAL = time[i-1]
        cVAL = time[i] 

        x = (cVAL - pVAL) * voltage[i]

        list_of_calculations.append(x)

    for i in range(len(list_of_calculations)):
        sum += list_of_calculations[i]

    # Normlize by constant 
    sum /= const

    # Calculate frequency
    frequency = sum / sim_length
    frequency /= 1e9
    if frequency > 1: 
        output_freq.insert(j, frequency)
    else:
        output_freq.insert(j, 0)

# Output the resulting information
# As an excel activation function
a = np.asarray([input_freq, output_freq])
np.savetxt("Activation_Function.csv", a, delimiter=",")
# As a plot to visualize 
plt.plot(input_freq, output_freq, label = "Found by Area", linewidth = '10')
plt.plot(input_freq, output_phase, label = "Found by Phase", linewidth = '5', linestyle = 'dashed')
plt.title("Input vs Output Frequency Relationships")
plt.xlabel("Input Frequencies (GHz)")
plt.ylabel("Output Frequencies (GHz)")
plt.show()

