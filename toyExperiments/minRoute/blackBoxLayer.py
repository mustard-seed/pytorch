import utils
import torch

class guassiandistanceBlackBox(Function):
    
    @staticmethod
    def forward(ctx, varMu, varSigma, endPoints):
        '''
        Generates waypoints based on the current variational parameters and compute distances
        
        Keyword arguments:
        ctx -- Context saver
        varMu -- torch.Tensor. m-by-n. 
                 m current variational means in the n-dimensional space
        varSigma --  torch.Tensor. m-by-1.
                     m current vartional variances
        endPoints -- torch.Tensor.
                     2-by-n. End points of the problem
        '''
        #Sanity check and obtain dimensinoality as well as number of movable points on a route
        assert (varMu.size()[0] == varSigma.size()[0]), "Number of movable points dervied from varMu differs that derived from varSigma!"
        assert (varMu.size()[1] == varSigma.size()[1]), "Number of dimenions derived from varMu differs that derived from varSigma!"
        m = varMu.size()[0]  #Number of movable points
        n = varMu.size()[1]  #Dimensions
        assert (endPoints.size()[0] == 2), "Number of end points does not equal to 2!"
        assert (endPoints.size()[1] == n), "Number of dimensions of the end points do not equal to that of the varMu or varSigma"
        
        route = torch.zeros(1,m+2,n)
        route[0][1:-1] = varMu
        route[0][0] = endPoints[0]
        route[0][-1] = endPoints[1]
        routeDistance = utils.l2Norm(routes)
        ctx.save_for_backward(varMu, varSigma)
        return routeDistance
        
    @staticmethod
    def backward(ctx, gradOfOutput)
        #Number of samples to generate for each point
        numSample = 20
        
        #Read back the saved tensors
        varMu, varSigma = ctx.saved_tensors
        m = varMu.size()[0]  #Number of movable points
        n = varMu.size()[1]  #Dimensions
        
        #Prepare route sampling vector
        routes = torch.zeros(numSample, m+2, n)
        #Generate numSample route samples
        for i in range (m):
            sampleMovablePoints = torch.normal(means=varMu, std=varSigma.pow(.5))
            routes[i][1:-1] = sampleMovablePoints
            #Place the end points
            routes[i][0] = endPoints[0]
            routes[i][-1] = endPoints[1]
        
        #Compute distance for each route
        routeDistances = utils.l2Norm(routes)
        
        ##TODO: implement the rest
