import sys
import os
# area under curve method
def numPeaksProto1(inputFile):
    op = open(inputFile)
    data = op.readlines()
    times = []
    nums = []
    goal = 2.07-15 # constant...should not need to change
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
    for i in range(1,len(times)):
        if start > end+diff*5: # account for noise. Once above constant, increase pulse count by 1
            pulseArea+=(nums[i]+nums[i-1])*(times[i]-times[i-1])/2
            if pulseArea >= goal:
                numPulses+=1
                startSpike.append(times[i])
        if pulseArea >= goal: # update end when noise > constant
            end = times[i]
        if nums[i] == 0: # set the start whenever amplitude is 0
            start = times[i]
            pulseArea = 0
    return (numPulses, startSpike)
# Hit certain amplitude method - structured the same way as numPeaksProto2
def numPeaksProto2(inputFile, maxAmplitude=0.0008):
    op = open(inputFile)
    data = op.readlines()
    times = []
    nums = []
    for i in data[1:]:
        pH = i.split(',')
        times.append(float(pH[0]))
        nums.append(float(pH[1]))
    numPulses = 0
    start = times[0]
    end = times[0]-1
    diff = times[1]-times[0]
    startSpike = []
    for i in range(1,len(times)):
        if start > end+diff*5:
            if nums[i] > maxAmplitude:
                numPulses+=1
                startSpike.append(times[i])
        if nums[i] > maxAmplitude:
            end = times[i]
        if nums[i] == 0:
            start = times[i]
    return (numPulses, startSpike)

def run():
    if len(sys.argv) == 1:
        print("""findPeak.py <countMethod> <fileName> <amplitudeThreshold>
countMethod=0 - find area under the curve. When the area hits the phi_0, a new pulse has been found
countmethod=1 - When amplitude gets above a certain amplitude, a new pulse has been found
amplitudeThreshold - Only used when countMethod=1. Set the amplitude. default to 0.0008""")
        return
    inputFile = sys.argv[2]
    if sys.argv[1] == '0':
        print(numPeaksProto1(inputFile))
    else:
        if len(sys.argv) > 3:
            print(numPeaksProto2(inputFile, float(sys.argv[3])))
        else:
            print(numPeaksProto2(inputFile))

run()
