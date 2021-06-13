"""
(ref) https://discuss.pytorch.org/t/custom-loss-functions/29387
"""
#%% 
import numpy as np 
import torch 
import torch.nn as nn 

#%% Reproducibility 
# (ref) https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(32)


#%%
def my_MSE_loss(output, target):
    SE = (output - target) ** 2
    MSE = torch.mean(SE)
    return MSE 




#%%
if __name__ == "__main__":
    model = nn.Linear(2, 2)  # Define a simple model  
                             # (ref) https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    x = torch.randn(1, 2)    # intput data 
    target = torch.randn(1, 2)  # label

    output = model(x) # inference 

    custom_loss = my_MSE_loss(output, target)
    torch_loss = nn.MSELoss()(output, target) # (ref) https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

    print(custom_loss)
    print(torch_loss)

#    loss.backward() 
#    print(model.weight.grad)
    

