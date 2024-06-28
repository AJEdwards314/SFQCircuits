import sys
import os
import matplotlib.pyplot as plt
import math
from scipy.signal import lfilter

# area under curve method
def numPeaksProto1(inputFile):
    op = open(inputFile)
    data = op.readlines()
    op.close()
    times = []
    nums = []
    goal = (2.07e-15)*.90 # constant...should not need to change
    # prepare arrays
    for i in data[1:]:
        pH = i.split(',')
        times.append(float(pH[0]))
        nums.append(float(pH[1]))
    numPulses = 0
    start = times[0]
    end = times[0]-1
    diff = times[1]-times[0]
    pulseArea = 0
    startSpike = []
    # find each pulse
    return math.floor(abs(nums[-1]*2/math.pi))
def modOutputs(inputFile, outputFile, startCur, endCur, precision):
    with open(inputFile, 'r') as file:
        data = file.read()
    x = []
    y = []
    for i in range(0, int((endCur-startCur)/precision)+1):
        i = startCur+precision*i
        newFile = data.format(i)
        f = open("{}-{}.cir".format(outputFile, i), "w")
        f.write(newFile)
        f.close()
        os.system("mkdir {}".format(outputFile))
        os.system("josim-cli {}-{}.cir -o {}//{}-{}.csv".format(outputFile, i, outputFile,outputFile,i))
        x.append(i)
        y.append(numPeaksProto1("{}//{}-{}.csv".format(outputFile,outputFile, i)))
    smoothingRate = 400
    plt.plot(x,y)
    plt.savefig("checkCurr")
    plt.close()
    b = [1.0 / smoothingRate]*smoothingRate
    a = 1
    y = lfilter(b,a,y)
    plt.plot(x,y)
    plt.savefig("checkCurr_denoise")
    plt.close()

def run():
    modOutputs(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))

run()
