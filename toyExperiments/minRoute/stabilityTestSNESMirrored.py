'''
stabilityTestSNESMirrored.py
Purpose: 
    Study the stability of the minimization algorithm at the optimal soluution, using antithetic sampling and SNES for gradient estimation
'''
import torch
import matplotlib.pyplot as plt
import time
import argparse
import os
import datetime
from utils import parameterGenerator
from minRouteModel import minRouteModel
import numpy as np

def main():
    #Load the following arguments from command line
    #m: int. Mandatory. Number of movable points
    #n: int. Mandatory. Dimension
    #s: int. Mandatory. Number of sampling points
    #iter: int. Mandatory. Number of iterations to run
    #logFlag: bool. Optional. Flag for enabling logging and not showing the plot
    parser = argparse.ArgumentParser(
        description="Generates data and plots for the (training) loss function values vs training \
        iteration number during a route minimization experiment.")
    parser.add_argument('--m', type=int, nargs='+', required=True,
        help='Number of movable points')
    parser.add_argument('--n', type=int, nargs='+', required=True,
        help='Number of dimensions')
    parser.add_argument('--s', type=int, nargs='+', required=True,
        help='Number of samples for gradient estimation')
    parser.add_argument('--iter', type=int, nargs=1, required=True,
        help='Number of training iterations per run')
    parser.add_argument('--logFlag', action='store_true',
        help='Optional flag for enabling logging and not showing the plot')
    parser.add_argument('--fitnessShapingFlag', action='store_true',
                        help='Optional flag for enabling fitness shaping')
    
    args = parser.parse_args()
    
    listM = args.m
    listN = args.n
    listS = args.s
    assert (len(listM) == len(listN) and len(listM) == len(listS)), "Size of lists are unequal!" 
    iterations = args.iter[0]
    logFlag = args.logFlag
    fitnessShapingFlag = args.fitnessShapingFlag
    numRun = 1
    
    #Optimizer settings
    learning_rate = 1e-2
    beta1 = 0.0
    beta2 = 0.5
    
   
    for index, m in enumerate(listM):
        outputDirectory = os.path.join(os.path.dirname(__file__),'stabilitySNESMirroredOutputs')
        if (not os.path.exists(outputDirectory)):
            os.makedirs(outputDirectory)
        currTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logFileName = os.path.join(outputDirectory, currTime+'.txt')
        logFile = None
        if logFlag:
            logFile = open(logFileName, "w")
        
        n = listN[index]
        s = listS[index]
        #Declare the end-points: origin and the [1,...,1] in R^n
        endPoints = torch.tensor([[-0.5] * n, [0.5] * n])
        endPoints.requires_grad_(False)
        #Declare m movable points in R^n
        points = np.transpose(np.tile( np.linspace(-0.5, 0.5, num=m, endpoint=False), (n,1)))
    
        varMuInitial = torch.tensor(points, dtype=torch.float)
        varSigmaInitial = torch.abs(varMuInitial)
    
    
        if logFlag:
            logFile.write("Adam setups: \n")
            logFile.write(("lr={lr}, Beta1 = {beta1}, Beta2 = {beta2}\n") \
                .format(lr=learning_rate, beta1=beta1, beta2=beta2))
            logFile.write(("m={m}, n={n}, numSample={s}, iterations={iter}\n") \
                .format(m=m, n=n, s=s, iter=iterations))
            logFile.write(("Initial varMu: {}\n") \
                .format(varMuInitial))
            logFile.write(("Initial varSigma: {}\n") \
                .format(varSigmaInitial))
    
        routeModel = minRouteModel(_endPoints = endPoints, _varMuInitial = varMuInitial, _varSigma2Initial = varSigmaInitial.pow(2.0))
    
        records = routeModel.minimize_route(numRun = numRun, \
            numIterations = iterations, numSamples = s, \
            learningRate = learning_rate, beta1 = beta1, beta2 = beta2, showProgress = True, requiresSigmaGrad = True, mirroredSampling = True, sNESOptimization=True, fitnessShapingFlag=fitnessShapingFlag)
    
        ax = records[1]
        logString = records[0]
    
        if logFlag:
            logFile.write(logString)
    
        ax.plot([0, iterations], [n, n], linewidth=2, linestyle='dashed', color='k')
        ax.set_title(("Antithetic and SNES sampling. " + "Fitness shaping: {ff} " + currTime+"\n m = {m}, n = {n}, s = {s}, lr={lr}, beta1={beta1}, beta2={beta2} \n optimize Mu: {om}, optimize Sigma: {os}") \
            .format(m=m, n=n, s = s, lr=learning_rate, beta1=beta1, beta2=beta2, om=True, os=True, ff=fitnessShapingFlag))
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        plt.savefig(fname="stabilitySNESMirroredOutputs/"+currTime+".png", format='png')
    
if __name__ == "__main__":
    main()

