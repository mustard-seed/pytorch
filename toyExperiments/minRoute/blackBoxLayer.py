import utils
import torch

class guassiandistanceBlackBox(torch.autograd.Function):
    
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
        
        # Bound the variances from below
        varSigma[varSigma<=0.0001] = 0.0001;
        
        route = torch.zeros(1,m+2,n)
        route[0][1:-1] = varMu
        route[0][0] = endPoints[0]
        route[0][-1] = endPoints[1]
        routeDistance = utils.routeL1(route)
        ctx.save_for_backward(varMu, varSigma, endPoints)
        return routeDistance
        
    @staticmethod
    def backward(ctx, gradOfOutput):
        '''
        Computes the gradient of the loss function with respect to each element of varMu and varSigma
        
        Keyword arguments:
        ctx -- Context saver containing the following tensors: varMu, varSigma, and endPoints
        gradOutput: Partial derivative of the loss function w.r.t. the distance. Scalar
        '''
        
        assert(gradOfOutput.size()[0] == 1), "Derviate of the loss function w.r.t. distance is not a scalar"
        #Number of samples to generate for each point
        numSample = 80
        
        #Read back the saved tensors
        varMu, varSigma, endPoints = ctx.saved_tensors
        m = varMu.size()[0]  #Number of movable points
        n = varMu.size()[1]  #Dimensions
        
        #Prepare route sampling vector
        routes = torch.zeros(numSample, m+2, n)
        #Generate numSample route samples
        #torch.manual_seed(1) #This is for deterministic results during debugging
        for i in range (numSample):
            sampleMovablePoints = torch.normal(mean=varMu, std=varSigma.pow(.5))
            routes[i][1:-1] = sampleMovablePoints
            #Place the end points
            routes[i][0] = endPoints[0]
            routes[i][-1] = endPoints[1]
        
        #Compute distance for each route
        routeDistances = utils.routeL1(routes)
        
        ##TODO: implement the gradient descent
        gradVarMu = gradVarSigma = gradEndPoints = None
        if ctx.needs_input_grad[0]:
            #If the means require gradient
            gradVarMu = torch.zeros(m,n)
            for k in range(m):
                for i in range(n):
                    for s in range(numSample):
                        gradVarMu[k][i] += 1/varSigma[k][i]*(routes[s][k+1][i] - varMu[k][i]) * routeDistances[s]
            gradVarMu = torch.div(gradVarMu, numSample)
            gradVarMu = torch.mul(gradVarMu, gradOfOutput)
        if ctx.needs_input_grad[1]:
            #If the variances require gradient
            gradVarSigma = torch.zeros(m,n)
            for k in range(m):
                for i in range(n):
                    for s in range(numSample):
                        gradVarSigma[k][i] += (-1*varSigma[k][i] + (routes[s][k+1][i] - varMu[k][i]).pow(2))*( (varSigma[k][i]).pow(-2) )/2 * routeDistances[s]
            gradVarSigma = torch.div(gradVarSigma, numSample)
            gradVarSigma = torch.mul(gradVarSigma, gradOfOutput)
        return gradVarMu, gradVarSigma, gradEndPoints
