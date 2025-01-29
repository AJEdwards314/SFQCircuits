# Code designed to make changes to circuit file and analyze specific parameters 

import pandas
import os 
import matplotlib.pyplot as plt
import numpy 

output_freq = []
output_phase = []

# Define points and increment value for frequency sweep 

# List will contain the points at which the resistive divider is set. 
# R1 is top resistor, and R2 is bottom resistor 
R1_val = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.3]
R2_val = 0.1

input_freq = [] 
increment = 10
max_freq = 200 
min_freq = 5 
step = int((max_freq - min_freq) / increment)
x = 0 

for i in range(step + 1):
    x = min_freq + ((i)*increment)
    input_freq.append(x)

# Rewrite in order to represent as a period value 

input_per = []

for i in range(len(input_freq)): 
    x = input_freq[i] * (1e9)
    x = (1 / x) * (1e12)
    x = "{:.5f}".format(x)
    input_per.append(x)

# This is where the iteration should begin, first importing the intended period to the circuit
# Next the os will run JOSIM and simulate under those specifications 
# Finally use the Flux_Detect in order to measure our output 

for k in range(len(R1_val)): 
    for j in range(len(input_per)): 

    # Function used for editing circuit file and importing the intended period for that run 

        def edit_circuit(): 

            with open("Neuron_Alex.cir") as fh: 
                str = fh.read()
                str = str.format(R1_val[k], R2_val, input_per[j])
                with open("SFQ_Neuron_TESTING.cir", 'w') as fh: 
                    fh.write(str)

    # Make changes to file before running

        edit_circuit() 

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

    a = numpy.asarray([input_freq, output_freq]).transpose()
    numpy.savetxt("Activation_Function.csv", a, delimiter=",")

    plt.plot(input_freq, output_freq, label = "Found by Area", linewidth = '10')
    plt.plot(input_freq, output_phase, label = "Found by Phase", linewidth = '5', linestyle = 'dashed')
    plt.title("Input vs Output Frequency Relationships")
    plt.xlabel("Input Frequencies (GHz)")
    plt.ylabel("Output Frequencies (GHz)")
    plt.savefig(f"R1 = {R1_val[k]}, R2 = 0.1.png")
    plt.clf()
    output_freq = []
    output_phase = [] 