import torch
import matplotlib as plt
import blackBoxLayer as bbLayer
import time

def parameterGenerator(parameters):
    '''
    Produce an iteratable list of parameters
    
    Keyword arguments:
    parameters: list of parameters
    '''
    for parameter in parameters:
        yield parameter

def main():
    #Declare 2 end points in R^2
    endPoints = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    #Declare 2 2D movable points
    varMu = ( torch.distributions.uniform.Uniform(torch.Tensor([2.0]), torch.Tensor([3.0])) ).sample([2,2]).squeeze()
    varSigma2 = torch.distributions.uniform.Uniform(torch.Tensor([0.1]), torch.Tensor([1.0]) ).sample([2,2]).squeeze()
    varMu.requires_grad_()
    varSigma2.requires_grad_()
    endPoints.requires_grad_(False)
    
    #Number of trainint iterations
    iterations = 2000
    #Instantiate an Adam optimizer
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(parameterGenerator([varMu, varSigma2]), lr=learning_rate, betas=(0.9,0.999))
    
    print ("Optimization begins.")
    timeBegin = time.time()
    for t in range(iterations):
        func = bbLayer.guassiandistanceBlackBox.apply
        
        #distance = func(varMu=varMu, varSigma=varSigma2, endPoints=endPoints) #Bad syntax. "apply() takes no keyword arguments"
        distance = func(varMu, varSigma2, endPoints)
        loss = distance.pow(2)
        
        optimizer.zero_grad()
        
        loss.backward()
        print(t, loss.item())
        print ("varMu: {}".format(varMu))
        print ("varSigma2: {}".format(varSigma2))
        print ("varMu.grad: {}".format(varMu.grad))
        
        optimizer.step()
        
    timeEnd = time.time()
    print ("Optimization ends. {} iterations elapsed.".format(iterations))
    print ("{} seconds elapsed.".format(timeEnd - timeBegin))

if __name__ == "__main__":
    main()
    
