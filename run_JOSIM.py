#########################################################################################################
# Author: Lucas Capone                                                                                  #
# Purpose: To run JOSIM circuit simulator and determine necessary parameters                            #
# Accepts: This code expects a circuit file in .cir form, and an input frequency / run time             #
# Outputs: The intention of this code is to produce graphs, and a .csv to visualize simulation results  #
# Additional Notes: This code is meant to function alongside research conducted in the NSC lab at UTD   #
#########################################################################################################

# Required libraries for code to function 
import pandas # Handles the .csv reading functions 
import os # Used to actually run JOSIM utilizing the operating system 
import matplotlib.pyplot as plt # For plotting the resulting simulation 
import numpy as np # Handles the random generation for poisson processes 

''' ---------------------------------------------------------------------
For this code we require the following functions: Poisson Generator, 
Circuit Editing, Startup, User Options, Run JOSIM, and Plotting. 
Poisson Generator: Takes a frequency and computes a pwl which can
be imported to a JOSIM circuit file in order to run 
Circuit Editing: This generates the output circuit file which JOSIM will 
run, and takes in the pwl(s), and input circuit file
Startup: This handles the input procedures and decides which mode and 
parameters will run 
Plotting: Final results will need to be displayed and saved for further
analysis 
User Options: This allows us to choose what parameters to run from the tests
and how we wish to plot them. For one plot alone, enter the specified test number, 
then a 0 (ex 1 0). For parallel voltage / phase testing, enter as a pair (ex 1 2)
Run JOSIM: Given the circuit file to run, simply exectutes JOSIM to generate results 
-----------------------------------------------------------------------''' 

def startup(): # Asks for user inputs 

    global frequency # These parameters are needed everywhere 
    global sim_length 
    global cir_file

    print("Run in default mode? (Y or N): ")
    mode = str(input())

    if(mode == 'N'): # User curated inputs 
        print("Circuit File (.cir): ")
        cir_file = str(input())
        print("Frequency at which to run: ")
        frequency = (float(input()) / 4)
        print("Simulation length (in pS): ")
        sim_length = float(input())
        sim_length *= 1e-12

    elif(mode == 'Y'): # If we perform many tests under one condition / debugging circuit
        cir_file = "conflu_neuron_v1.cir"
        frequency =  50
        sim_length = 10000e-12 

def poisson(freq, seed, sim): # Creates the poisson distributed spike train 
    sum = 0 # Keeps track of summing when making time list 
    delta = 10e-12 # Half of the spike base length 
    const = 2.0678e-15 # SFQ constant
    height = (const * 2) / (2 * delta) # Determines spike height (amplitude)
    time = [] # List will be populated with spike times 
    x = [] # The list corresponing to x-axis 
    vin = [] # The list containing all amplitudes 
    c = 0 # used for correcting errors with spike generation 

    def generate_poisson_spikes(rate, gen): # Generates the guassian randomness 
        spike = gen.exponential(1/rate)
        return spike

    rate = freq * 1e9  # Average rate (lambda) of the Poisson distribution
    gen = np.random.default_rng(seed) # Takes in the given seed in case of multiple pwl's

    # This must intentionally overshoot slightly to give JOSIM enough data points
    while(sum <= sim): # Generate list of times. 
        sum += generate_poisson_spikes(rate,gen)
        time.append(sum)

    # look at difference in times between values. if the difference in the spike points (peaks) 
    # is less than 2 base length, then correct that spike time

    for i in range(1, len(time)): # observe and determine if correction is needed 
        c = time[i] - time[i-1]   # compute difference 
        if(c < 2 * delta):        # check if needed 
            correction = (2 * delta) - c # final correction term 
            time[i] += correction # shift in time to make difference 0 at worst 

    for i in range(len(time)):  # Generates the spike occurnces 
        if i == 0:              # JOSIM requires first point be 0 
            x.append(0)
            vin.append(0)
        else:                   # The rest is populated as normal 
            x.append(time[i - 1] - delta)
            vin.append(0)       # 0 at the ends of spike
            x.append(time[i - 1])
            vin.append(height)  # Amplitude at center of spike
            x.append(time[i - 1] + delta)
            vin.append(0)

    pwl = "" # String which is then imported to JOSIM

    for i in range(len(x)): # Final construction of pwl 
        pwl += str((x[i])) + " " + str(round(vin[i], 5)) + " "

    return pwl # Return the resulting string to the code 

def edit_circuit(cir, pwl1, pwl2, pwl3, pwl4, pwl5, R2): # Takes the needed pwl's and generates the circuit file 
    with open(f"{cir}") as fh: 
        str = fh.read() # Read the circuit file and save 
        str = str.format(R2, pwl1, pwl2, pwl3, pwl4, pwl5) # Make any necessary changes 
        with open("SFQ_Neuron_TESTING.cir", 'w') as fh: # Write in new changes to circuit 
            fh.write(str)

def run_josim(): # Call JOSIM and exectute statement
    os.system('josim-cli -o ./SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv SFQ_Neuron_TESTING.cir')

def create_sweep(): 

    # ins = [] 

    #print("Max Freq (GHz): Min Freq(GHz): Step: ")

    # for i in range(3): 
    #     ins.append(float(input())) 

    #if(ins[1] == 0): # Correct thiSs term if the user sets it to 0 
    #    ins[1] += 0.001

    sweeplist = [] 

    #max = ins[0] 

    # sum = 0 

    # i = 0 

    # while(sum < max):

    #     sum = ins[1] + ( i * ins[2])

    #     i += 1 

    #     sweeplist.append(sum)
    
    # sweeplist  = [0.001, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62.5, 65, 67.5, 70, 75, 80]
    sweeplist = [0.001, 0.021, 0.041, 0.061, 0.081, 0.101, 0.121, 0.141, 0.161, 0.181, 0.201, 0.221, 0.241, 0.261, 0.281, 0.301, 0.321, 0.341, 0.361, 0.381, 0.401, 0.421, 0.441, 0.461, 0.481, 0.501, 0.521, 0.541, 0.561, 0.581, 0.601, 0.621, 0.641, 0.661, 0.681, 0.701, 0.721, 0.741, 0.761, 0.781, 0.801, 0.821, 0.841, 0.861, 0.881, 0.901, 0.921, 0.941, 0.961, 0.981, 1]
    
    return sweeplist 

def print_options(data): # Function will allow user to choose what to print 
    # Due to these being shared for the next portion, it is simpler to initialize globally 
    global num_test
    global name_row
    num_test = 0 # Will ask user how many graphs are needed  
    name_row = data.columns.tolist() # This will give back all of the column names 

    print("The avaliable parameters which can be exported are: ") 

    for i in range(1, len(name_row)): # Print avaliable options 
        print(f"{i} - {name_row[i]}")

    print("How many needed tests?: ") # Ask for number of needed tests
    num_test = int(input()) 

    rows, cols = (2, num_test)
    tests = [[0 for i in range(cols)] for j in range(rows)]

    print("Specify which to plot in pairs (ex. 1 2): ") # Ask which need plotting 
    for i in range(num_test): 
        user = int(input()) 
        tests[0][i] = user 
        user = int(input()) 
        tests[1][i] = user 

    return tests # Give back the tests that need plotting 

def plot(tests, names, data): # Takes our recieved data and determines what to plot

    time = data['time'].tolist() # time base, used regardless

    for i in range(num_test): # Find the elements to plot 
        x = tests[0][i]
        y = tests[1][i]

        if(y == 0): # Case where data is plotted alone 
            
            data1 = data[f"{names[x]}"].tolist() # get the first data point to plot 
        
            plt.plot(time, data1, linewidth = '1') # Plot requested data alone 
            plt.title(f"{names[x]}")
            plt.xlabel("Time")
            plt.ylabel(f"{names[x]} Magnitude (V or Phase)")
            plt.plot()
            plt.savefig(f"Output {x}.png") # Save under column name 
            plt.clf()

        if(y > 0): # In the case where the user does want to plot together 

            data1 = data[f"{names[x]}"].tolist() # get the first data point to plot
            data2 = data[f"{names[y]}"].tolist() # get the first data point to plot 

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6)) # Plot requested data together

            # Plot data in each subplot
            ax1.plot(time, data1, label = f"{names[x]}", color = "blue") # First Point 
            ax1.set_title(f"{names[x]}")
            ax1.legend()
            ax1.grid()

            ax2.plot(time, data2, label = f"{names[y]}", color = "red") # Second Point 
            ax2.set_title(f"{names[y]}")
            ax2.legend()
            ax2.grid()

            plt.savefig(f"{names[x]}.png".replace('|', '-')) # Save under column name 
            plt.clf()

def in_out_rate(data, points): 

    rate1 = data[f'{points[0]}'].tolist() # Capture final count for both inputs
    rate2 = data[f'{points[1]}'].tolist()
    rate3 = data[f'{points[2]}'].tolist() 
    rate4 = data[f'{points[3]}'].tolist() 
    rate5 = data[f'{points[4]}'].tolist() # and for the output 

    xplot.append( ((rate1[-1] + rate2[-1] + rate3[-1] + rate4[-1]) / (sim_length * 1e9 * 2 * pi) ) )
    yplot.append( rate5[-1] / (2 * pi) )

    return xplot, yplot 

def test_plot(plot1, plot2, title, mode): # Function will plot other kinds of data (Sweeps / ETC) 


    if(mode == 1): 

        plt.scatter(plot1, plot2) # Plot requested data alongside each other  
        plt.title(f"{title}")
        plt.xlabel("Given Frequencies")
        plt.ylabel("Resulting Frequencies")
        plt.plot()
        plt.gca().set_ylim(bottom = 0)
        plt.gca().set_xlim(left = 0)
        plt.savefig(f"{title}.png") # Save under column name 
        plt.clf()

    if(mode == 2):
        plt.scatter(plot1, plot2) # Plot requested data alongside each other  
        plt.title(f"{title}")
        plt.xlabel("Input Frequencies")
        plt.ylabel("Output Frequencies")
        plt.plot()
        plt.gca().set_ylim(bottom = 0)
        plt.gca().set_xlim(left = 0)
        plt.savefig(f"{title}.png") # Save under column name 
        plt.clf()

def freq_sweep(data): # Function also performs a sweep, in a frequency respose fashion 
    output = data['P(B0L|X06)'].tolist() 
    phase_max = (output[-1] / (sim_length * 1e9 * 2 * pi)) * (-1)

    return phase_max 
    
# Main area of code where all of the functions are called in their respective order 

pi = 3.14159 # value of pi to determine number of phase rotations

R2_val = [0.1] # default value for R2 

startup() # Get user Values 

print("Run Standard Test? (Y or N): ") # Determine the kind of tests that are needed 
standard = str(input()) 

if(standard == 'Y'): # Standard test for transient response 

    pwl1 = poisson(frequency, 0, sim_length) # Create the needed pwl(s)
    pwl2 = poisson(frequency, 1, sim_length) # Create the needed pwl(s)
    pwl3 = poisson(frequency, 2, sim_length) # Create the needed pwl(s)
    pwl4 = poisson(frequency, 3, sim_length) # Create the needed pwl(s)
    pwl5 = poisson(50, 4, sim_length) # Create the pwl for the inhibitory input 

    edit_circuit(cir_file, pwl1, pwl2, pwl3, pwl4, pwl5, R2_val[0]) # Function takes all needed pwl's and file, to send to JOSIM

    if __name__ == "__main__":  # Run JOSIM 
        run_josim()

    data = pandas.read_csv('SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv') # Reads the generated CSV and saves 

    tests = print_options(data) # Get the contents of tests from the user 

    print(name_row)

    plot(tests, name_row, data) # Finally plot outputs and determine 

print("Run parameter sweep? (Y or N): ") # Determine if sweep of resistor value is needed 
paramsweep = str(input()) 

if (paramsweep == 'N'): # In no case set value once
    R2_val = [0.1]

elif(paramsweep == 'Y'): # In yes case values sweep based on following list 
    R2_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

print("Run in out rate test? (Y or N): ")
in_out_test = str(input()) # Ask for I/O test

print(len(R2_val))

for k in range(len(R2_val)): # Sweeps the runs for all values of R2 

    print(k)

    if (in_out_test == 'Y'): # If test is accepted 

        # Set to test Confluence alone 

        in_out_data = ['P(B05|X01)', 'P(B05|X02)', 'P(B05|X03)', 'P(B05|X04)', 'P(B0L|X06)'] # Which points to measure in sweep 

        xplot = [] # Create list for the x points 

        yplot = [] # Create list for y points

        # for i in range(3): 
        #    in_out_data[i] = str(input(f"Enter Point Number {i}: ")) # Get inputs for which points to measure 

        sweeplist = create_sweep() # Get parameters for the sweep 

        output_freq = []

        for i in range(len(sweeplist)): # Begin the frequency sweep 

            pwl1 = poisson(sweeplist[i], 0, sim_length)
            pwl2 = poisson(sweeplist[i], 1, sim_length)
            pwl3 = poisson(sweeplist[i], 2, sim_length)
            pwl4 = poisson(sweeplist[i], 3, sim_length)
            pwl5 = poisson(50, 4, sim_length) # Create the pwl for the inhibitory input 
            
            edit_circuit(cir_file, pwl1, pwl2, pwl3, pwl4, pwl5, R2_val[k]) 

            run_josim()

            data = pandas.read_csv('SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv') # Reads the generated CSV and saves

            # This list saves frequency sweep data 

            output_freq.append(freq_sweep(data)) 

            xplot , yplot = in_out_rate(data, in_out_data)
        
        test_plot(xplot, output_freq, f"Neuron Frequency Trials - LOW RANGE - (R1 = 0.5 R2 = {R2_val[k]})", 2) # plots the output frequency sweep    