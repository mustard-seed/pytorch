import torch
import matplotlib.pyplot as plt
import blackBoxLayer as bbLayer
import blackBoxLayerMirrored as bbLayerMirrored
from normalLayer import normalLayer
import nesLayerMirrored
from utils import parameterGenerator
import io

class minRouteModel:
    """
    A route minimization problem
    """
    
    def __init__(self, \
        _endPoints, _varMuInitial, _varSigma2Initial):
        """
        Initializes the end points, coordinate means, and variance mean
        """
        assert (_varMuInitial.size()[0] == _varSigma2Initial.size()[0]), "Number of movable points dervied from the variational means differ from those derived from the variational variances!"
        assert (_varMuInitial.size()[1] == _varSigma2Initial.size()[1]), "Number of dimenionsdervied from the variational means differ from those derived from the variational variances!"
        self.numMovablePoints = _varMuInitial.size()[0]  #Number of movable points
        self.numDimensions = _varSigma2Initial.size()[1]  #Dimensions
        assert (_endPoints.size()[0] == 2), "Number of end points does not equal to 2!"
        assert (_endPoints.size()[1] == self.numDimensions), "umber of dimensions of the end points are not consistent with those of the vartional parameters!" 
        
        self.varMuInitial = _varMuInitial
        self.varSigma2Initial = _varSigma2Initial
        self.endPoints = _endPoints
    
    def minimize_route(self, \
        numRun, numIterations, numSamples, \
        learningRate, beta1, beta2, \
        showProgress = False, \
        requiresMuGrad = True,
        requiresSigmaGrad = True,
        mirroredSampling = False,
        normalOptimization=False,
        sNESOptimization=False,
        fitnessShapingFlag=False,
        plotLog = False):
        """
        Minimize the route
        
        Args:
            numRun (int): Number of training runs to perform using the same initialization
            numIterations (int): Number of training iterations per run
            numSamples (int): Number of sampling for gradient estimation
            learningRate (float): Learnig rate for the Adam optimizer
            beta1 (float): Adam optimizer parameter
            beta2 (float): Adam optimizer parameter
            showProgress (bool): Flag to printing progress to console
        
        Returns:
            The return value (list). [0] is a logger string, and [1] is the pyplot plot
        """
        
        logFile = io.StringIO()
        
        # Drawing settings
        colors = ['r', 'g', 'b', 'c', 'm', 'k']
        fig, ax = plt.subplots()
        
        for runs in range(numRun):
            listIter = []
            listLoss = []
            varMu = torch.Tensor(self.varMuInitial.clone())
            if requiresMuGrad:
                varMu.requires_grad_()
            varSigma2 = torch.Tensor(self.varSigma2Initial.clone())
            varSigma = torch.tensor(torch.pow(self.varSigma2Initial, 0.5).clone())
            if requiresSigmaGrad:
                varSigma2.requires_grad_()
                varSigma.requires_grad_()
            if (normalOptimization):
                optimizer = torch.optim.Adam(parameterGenerator([varMu]), lr=learningRate, betas=(beta1,beta2))
            elif (sNESOptimization):
                optimizer = torch.optim.Adam(parameterGenerator([varMu, varSigma]), lr=learningRate, betas=(beta1, beta2))
            else:
                optimizer = torch.optim.Adam(parameterGenerator([varMu, varSigma2]), lr=learningRate, betas=(beta1,beta2))
            logFile.write("===========Run: {}================\n".format(runs))
            for t in range (numIterations):
                func = None
                # Create an alias for the apply function
                if (normalOptimization == True):
                    func = normalLayer
                elif (sNESOptimization == True):
                    func = nesLayerMirrored.nesLayerMirrored.apply
                elif (mirroredSampling == False):
                    func = bbLayer.guassiandistanceBlackBox.apply
                else:
                    func = bbLayerMirrored.guassiandistanceBlackBoxMirrored.apply
                
                
                #Forward pass
                if (normalOptimization == True):
                    distance = func(varMu, self.endPoints)
                elif (sNESOptimization == True):
                    distance = func(varMu, varSigma, self.endPoints, numSamples, fitnessShapingFlag)
                else:
                    distance = func(varMu, varSigma2, self.endPoints, numSamples)
                loss = distance.pow(2)
                listIter.append(t)
                listLoss.append(loss.item())
                logFile.write(("Iter = {t}, loss={loss}\n") \
                    .format(t=t, loss=loss.item()))
                logFile.write (("varMu: {}\n").format(varMu))
                if (normalOptimization == False and sNESOptimization == False):
                    logFile.write (("varSigma2: {}\n").format(varSigma2))
                elif (normalOptimization == False):
                    logFile.write(("varSigma2: {}\n").format(varSigma))
                logFile.write (("varMu.grad: {}\n").format(varMu.grad))
                
                #Prints progress to console periodically
                if ( int(t) % int(100) == 0 and showProgress):
                    print ( ("Run: {r}, iter: {iter}\n") \
                        .format(r=runs, iter=t))
                    print (("loss={}\n").format(loss.item()))
                
                #Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (plotLog):
                ax.semilogy (listIter, listLoss, color=colors[runs], linewidth=2, linestyle='solid')
            else:
                ax.plot(listIter, listLoss, color=colors[runs], linewidth=2, linestyle='solid')
        logString = logFile.getvalue()
        records = [logString, ax]
        logFile.close()
        return records
        

