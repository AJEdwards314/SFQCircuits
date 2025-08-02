#########################################################################################################
# Author: Lucas Capone                                                                                  #
# Purpose: To run JOSIM circuit simulator on all of the NSC subcircuits                                 #
# Accepts: This code expects a circuit file in .cir form, and allows user to pick input parameters      #
# Outputs: The intention of this code is to produce graphs, and a .csv to visualize simulation results  #
# Additional Notes: This code is meant to function alongside research conducted in the NSC lab at UTD   #
#########################################################################################################

# Required libraries for code to function 
import pandas # Handles the .csv reading functions 
import os # Used to actually run JOSIM utilizing the operating system 
import matplotlib.pyplot as plt # For plotting the resulting simulation 
import numpy as np # Handles the random generation for poisson processes 

''' ---------------------------------------------------------------------
This code takes the previously designed run_JOSIM and rewrites to 
improve upon certain elements and test the new SUBCIRCUITS file. 
This code can now read the sim length from the provided file instead of
having the user provide it manually. 
The addition of descriptiors for the outputs in order to more 
effectively visualize them has been added as well. 
Given the cir file this feature will read the values and associate 
(and generate) more effective names for the output file names
-----------------------------------------------------------------------''' 

def startup(): # Asks for user inputs 

    frequency = 5 # These parameters are needed everywhere 
    cir_file = 'SUBCIRCUITS.cir'

    search = '.tran'
    cline = []

    with open(f'{cir_file}', 'r') as file: 
        for line in file: 
            if search in line: 
                cline = line.split()
                x = cline[2].strip('p') # This should save the value for length of the simulation
                x = int(x)

    sim_length = x * (1e-12) # Corrects simlength to be in picoseconds 

    return frequency, sim_length, cir_file 

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

def edit_circuit(cir, pwl1, pwl2): # Takes the needed pwl's and generates the circuit file 
    with open(f"{cir}") as fh: 
        str = fh.read() # Read the circuit file and save 
        str = str.format(pwl1, pwl2) # Make any necessary changes 
        with open("SFQ_Neuron_TESTING.cir", 'w') as fh: # Write in new changes to circuit 
            fh.write(str)

def run_josim(): # Call JOSIM and exectute statement

    if __name__ == "__main__":  # Run JOSIM 
        os.system('josim-cli -o ./SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv SFQ_Neuron_TESTING.cir')

def print_options(data): # Function will allow user to choose what to print 

    # Due to these being shared for the next portion, it is simpler to initialize globally 

    global num_test

    num_test = 0 # Will ask user how many graphs are needed  
    name_row = data.columns.tolist() # This will give back all of the column names 

    print("The avaliable parameters which can be exported are: ") 

    for i in range(1, len(name_row)): # Print avaliable options 
        print(f"{i} - {name_row[i]}")

    print("Output all?: (Y or N)") # Ask for number of needed tests
    slct1 = (input()) 

    if slct1 == 'Y': 
        num_test = len(name_row)
        list_one = []
        list_two = []

        for i in range(num_test): 
            list_one.append(i)
            list_two.append(0)

        tests = [list_one , list_two]

    elif slct1 == 'N' : 
        rows, cols = (2, num_test)
        tests = [[0 for i in range(cols)] for j in range(rows)]
        #print("Specify which to plot in pairs (ex. 1 2): ") # Ask which need plotting 
        for i in range(num_test): 
            user = int(input()) 
            tests[0][i] = user 
            user = int(input()) 
            tests[1][i] = user

    return tests # Give back the tests that need plotting 

def plot(tests, names, new_name, data): # Takes our recieved data and determines what to plot

    time = data['time'].tolist() # time base, used regardless

    for i in range(num_test): # Find the elements to plot 
        x = tests[0][i]
        y = tests[1][i]

        if(y == 0): # Case where data is plotted alone 
            
            data1 = data[f"{names[x]}"].tolist() # get the first data point to plot 
        
            plt.plot(time, data1, linewidth = '1') # Plot requested data alone 
            plt.title(f"{new_name[x]}")
            plt.xlabel("Time")
            plt.ylabel(f"{new_name[x]} Magnitude (V or Phase)")
            plt.plot()

            # List of names for the output plots 

            plt.savefig(f"{new_name[x]}.png") # Save under column name 
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

def text_search(cir): # Runs through file to find what each output should be named 

    search_for = []
    output_nodes = []
    output_names = []
    print_node = [] 
    print_names = []

    # Run through the text file and find all of the subcircuit names in their corresponding lines 
    # With those lines, append them to output list in order to serve as the output names for the files 

    begin = 'MAIN'
    end = 'OUTPUTS'
    marker = 0
    start = 0 
    finish = 0
    new_file = []

    with open(f"{cir}", 'r') as file2: # First run gives us the start and end position of the node names 
        for line in file2:
            if begin in line:
                start = marker
            if end in line: 
                finish = marker
            marker += 1 

        file2.close()

    marker = 0 

    with open(f"{cir}", 'r') as file2: # Second run saves the needed information in seperate list 
        for line in file2:
            if marker >= start and marker <= finish and len(line) < 25 and len(line) > 2: 
                temp = line.strip()
                temp = line.strip('\n')
                new_file.append(temp)
            marker += 1 

        file2.close()

    # At this point the inputs and subcircuits / main components are isolated. 
    # From here the operation just needs to group the name of the node, with the node itself 
    # Only save the output node for the subcircuits 

    for i in range(len(new_file)): 
        cline = [] # cline stands for current line
        cline = new_file[i].split()
        string1 = cline[0] # save the first word
        char1 = string1[0] # save the first character 
        match char1: 
            case 'V': # (VINX NODE1 NODE2 TYPE)
               output_names.append(cline[0])
               output_nodes.append(cline[1])
            case 'X': # X0X SUBCIRCUIT NODE1 NODE2 
               temp = cline[0] + '-' + cline[1]
               output_names.append(temp)
               output_nodes.append(cline[-1])
            case 'B': # B0X NODE1 NODE2 MODEL AREA
               output_names.append(cline[0])
               temp = cline[1] + ' - ' + cline[2]
               output_nodes.append(temp)
            case 'R': # R0X NODE1 NODE2 VALUE
                output_names.append(cline[0])
                temp = cline[1] + ' - ' + cline[2]
                output_nodes.append(temp)

    # This performs similar operation but instead looks for the .print statement to find output nodes  

    with open(f"{cir}", 'r') as file2: 
        search_for = '.print'
        for line in file2:
            if search_for in line:
                new_line = line.split()
                for i in range(len(new_line)): 
                    if len(new_line[i]) < 3:
                        print_node.append(new_line[i])
                    elif len(new_line[i]) > 3 and new_line[i] != search_for: 
                        print_names.append(new_line[i]) 
        file2.close()

    # At this point the lists are made, but just need to be adjusted in order to fit

    pos2neg = ['7', '8'] # These are the nodes for the +/i converter 

    i = 0

    # This is for all (if any) of the circuits that need multiple inputs 
    while (i < len(output_names)):
        if 'P&N' in output_names[i]: 
            output_names.insert(i, output_names[i])
            output_nodes.insert(i, pos2neg[0])
            i = len(output_names)
        else: 
            i += 1

    return output_names, output_nodes

def data_output(data): # This generates the names and data organization

    col_name = data.columns.tolist() # This will give back all of the column names

    col_name_new = [] # This is for generating the new column names with proper formating 

    for i in range(len(col_name)):
        temp = col_name[i].replace("(", " ").replace(")", "").replace("|", "-")
        col_name_new.append(temp)

    col_name_new.pop(0) # This removes the 'time' from the list of titles 

    node_name = [] # These will hold the name or type of the node 
    node_nums = [] # These will be the number of the node 

    for i in range(len(col_name_new)): # This makes list for the number, and the node of each article

        temp = []
        temp = col_name_new[i].split() 
        
        node_name.append(temp[0])
        node_nums.append(temp[1])

    # Iterate through the column names and find which name corresponds to each value 
    # This next step should eventually be a function rather than this for loop 

    ref_list = [] # This list should replace the col_names list with the proper names 

    for i in range(len(node_nums)): 
        cnode = node_nums[i] # current node being observed 
        for k in range(len(names)):
            if cnode == nodes[k]: # This is for most of the nodes 
                ref_list.append(f'{names[k]}-{nodes[k]}-{node_name[i]}')
            elif len(cnode) == 3 and cnode == names[k]: 
                ref_list.append(f'{cnode}-{nodes[k]}-{node_name[i]}')
            elif len(cnode) == 7 and cnode != names[k] and k == len(names) - 1: 
                ref_list.append(f'{cnode}-{node_name[i]}')

    # Correction term is needed to compensate for the fact that the P&N circuit (like many others perhaps)
    # has multiple inputs 

    ref_list.insert(0, 'time') # replaces the list for names in the same format but with the proper correlations to the nodes

    return ref_list, col_name # Final output in form of updated list 

#####################################################################################
# MAIN - area of code where all of the functions are called in their respective order 
#####################################################################################

pi = 3.14159 # value of pi to determine number of phase rotations

frequency, sim_length, cir_file = startup() # Get user values and circuit file 

pwl1 = poisson(10, 0, sim_length) # Generate PWL input based on provided frequency in startup 
pwl2 = poisson(frequency, 1, sim_length) # Generate 2nd pwl for the other input to the circuit 

edit_circuit(cir_file, pwl1, pwl2) # Places both pwls into the circuit file and rewrites to new executable file

run_josim() # Runs josim with the new file 

data = pandas.read_csv('SFQ_Neuron_TESTING/SFQ_Neuron_TESTING.csv') # Reads the generated CSV and saves 

names, nodes = text_search(cir_file) # Provides some needed data for improved output names

ref_list, col_name = data_output(data)

tests = print_options(data) # Get the contents of tests from the user 

plot(tests, col_name, ref_list, data) # Plot all of the data or chosen parameters


    