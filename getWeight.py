import sys
import os
import matplotlib.pyplot as plt
import math
from scipy.signal import lfilter
import pandas as pd

# area under curve method
def numPeaksProto1(inputFile):
    op = open(inputFile)
    data = op.readlines()
    op.close()
    pH = data[-1].split(',')
    # find each pulse
    return float(pH[3])/float(pH[2])

def modOutputs(inputFile, outputFile, startCur, endCur, precision):
    with open(inputFile, 'r') as file:
        data = file.read()
    x = []
    y = []
    y_area = []
    os.system("mkdir outputCsvs".format(outputFile))
    for i in range(0, int((endCur-startCur)/precision)+1):
        i = startCur+precision*i
        newFile = data.format(i,0-i)
        f = open("outputCircuits//{}-{}.cir".format(outputFile, i), "w")
        f.write(newFile)
        f.close()
        os.system("josim-cli outputCircuits//{}-{}.cir -o outputCsvs//{}-{}.csv".format(outputFile, i,outputFile,i))
        x.append(i)
        y.append(numPeaksProto1("outputCsvs//{}-{}.csv".format(outputFile, i)))
        os.system('rm outputCsvs//{}-{}.csv'.format(outputFile, i))
    smoothingRate = 1000
    b = [1.0 / smoothingRate]*smoothingRate
    a = 1
    plt.plot(x,y)
    plt.savefig("checkCurr_area")
    plt.close()
    y_area = lfilter(b,a,y)
    plt.plot(x,y)
    plt.savefig("checkCurr_area_denoise")
    plt.close()
    data = {'input current': x, 'output probabiility': y}
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)
    

def run():
    os.system("mkdir outputCircuits")
    modOutputs(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))

run()
