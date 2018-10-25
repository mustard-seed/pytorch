'''
scalabilityTest.py
Purpose: 
    Generate data and plots for the (training) loss function values vs training
    iteration number during a route minimization experiment.
    The dimensionality, number of movable points are configurable via commandline
    The fixed points are origin and [1.0, ..., 1.0]
    The movable points are initialized the same
'''
import torch
import matplotlib.pyplot as plt
import blackBoxLayer as bbLayer
import time
import argparse
import os
import datetime
from utils import parameterGenerator
from minRouteModel import minRouteModel

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
    parser.add_argument('--m', type=int, nargs=1, required=True,
        help='Number of movable points')
    parser.add_argument('--n', type=int, nargs=1, required=True,
        help='Number of dimensions')
    parser.add_argument('--s', type=int, nargs=1, required=True,
        help='Number of samples for gradient estimation')
    parser.add_argument('--iter', type=int, nargs=1, required=True,
        help='Number of training iterations per run')
    parser.add_argument('--logFlag', action='store_true',
        help='Optional flag for enabling logging and not showing the plot')
    parser.add_argument('--constVariance', action='store_true',
        help='Optional flag for optimizing the means only')
    parser.add_argument('--mirroredSampling', action='store_true',
        help='Optional flag for mirrored Sampling')
    
    args = parser.parse_args()
    
    m = args.m[0]
    n = args.n[0]
    numSample = args.s[0]
    iterations = args.iter[0]
    logFlag = args.logFlag
    numRun = 5
    requiresVarGrad = not args.constVariance
    mirroredSampling = args.mirroredSampling
    
    currTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logFileName = os.path.join(os.path.dirname(__file__),'scalabilityOutputs/'+currTime)
    logFile = None
    if logFlag:
        logFile = open(logFileName, "w")
    
    # Drawing settings
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    
    #Optimizer settings
    learning_rate = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    
    
    #Declare the end-points: origin and the [1,...,1] in R^n
    endPoints = torch.tensor([[0.0] * n, [1.0] * n])
    endPoints.requires_grad_(False)
    #Declare m movable points in R^n
    varMuInitial = ( torch.distributions.uniform.Uniform(torch.Tensor([-10.0]), torch.Tensor([10.0])) ).sample([m,n]).squeeze()
    varSigma2Initial = torch.distributions.uniform.Uniform(torch.Tensor([0.1]), torch.Tensor([2.0]) ).sample([m,n]).squeeze()
    
    if logFlag:
        logFile.write("Adam setups: \n")
        logFile.write(("lr={lr}, Beta1 = {beta1}, Beta2 = {beta2}\n") \
            .format(lr=learning_rate, beta1=beta1, beta2=beta2))
        logFile.write(("m={m}, n={n}, numSample={s}, iterations={iter}\n") \
            .format(m=m, n=n, s=numSample, iter=iterations))
        logFile.write(("Initial varMu: {}\n") \
            .format(varMuInitial))
        logFile.write(("Initial varSigma2: {}\n") \
            .format(varSigma2Initial))
        logFile.write(("Mirrored Sampling: {}\n") \
            .format(mirroredSampling))
        logFile.write(("Fixed Variances: {}\n") \
            .format(args.constVariance))
    
    routeModel = minRouteModel(_endPoints = endPoints, _varMuInitial = varMuInitial, _varSigma2Initial = varSigma2Initial)
    
    records = routeModel.minimize_route(numRun = 5, \
        numIterations = iterations, numSamples = numSample, \
        learningRate = learning_rate, beta1 = beta1, beta2 = beta2, showProgress = True, \
        requiresSigmaGrad = requiresVarGrad, mirroredSampling = mirroredSampling)
    
    ax = records[1]
    logString = records[0]
    
    if logFlag:
        logFile.write(logString)
    
    ax.plot([0, iterations], [n, n], linewidth=2, linestyle='dashed', color='k')
    ax.set_title( ("Scalability Test \n" + currTime+"\n m = {m}, n = {n}, s = {s}, lr={lr}, beta1={beta1}, beta2={beta2} \n" + \
        "Constant variances: {constVar}. Mirrored Sampling: {mirroredSampling}") \
        .format(m=m, n=n, s = numSample, lr=learning_rate, beta1=beta1, beta2=beta2, constVar = args.constVariance, mirroredSampling=mirroredSampling))
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    if logFlag:
        plt.savefig(fname="scalabilityOutputs/"+currTime+".png", format='png')
    else:
        plt.show()
    
if __name__ == "__main__":
    main()

