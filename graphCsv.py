import sys

import matplotlib.pyplot as plt
import os
def graph(inputFile):
    op = open("./"+inputFile+"/"+inputFile+".csv")
    data = op.readlines()
    colNames = data[0].split(',"')
    nums = dict()
    for i in range(len(colNames)):
        colNames[i] = colNames[i].replace('"','')
        colNames[i] = colNames[i].replace('\n', '')
        nums[colNames[i]] = []
    data = data[1:]
    for i in range(len(data)):
        pH=(data[i].split(','))
        for j in range(len(pH)):
            pH[j] = float(pH[j])
            nums[colNames[j]].append(pH[j])
    for i in colNames[1:]:
        plt.plot(nums['time'], nums[i], label=i)
        plt.title(i)
        plt.legend()
        plt.savefig("./"+inputFile+"/"+i)
        plt.close()
    for i in colNames[1:]:
        plt.plot(nums['time'], nums[i], label=i)
    plt.title("everything together (scale may make graph bad)")
    plt.legend()
    plt.savefig("./"+inputFile+"/"+inputFile+".png")


def run():
    inputFile = sys.argv[1]
    os.system("mkdir " + inputFile)
    os.system("josim-cli " + inputFile + ".cir -o ./" + inputFile + "/" + inputFile+".csv")
    graph(inputFile)

run()