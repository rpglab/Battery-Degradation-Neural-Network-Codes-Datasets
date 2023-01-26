'''
Neural Network for 1 Dimentional Data
'''
import torch
from torch import nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
#from readtestdata_size import *
#from readcsv import *
#from readdata100_95 import *
from nn_class import Net
import os
from sklearn.model_selection import cross_val_score, train_test_split
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# load data
x_sample =np.load('x_sample.npy')
y_sample =np.load('y_sample.npy')
x_test =np.load('x_test.npy')
y_test =np.load('y_test.npy')

###parameter setup####
feature_num = 5
print_every = 10000000000
BATCH_SIZE = 256
epoches = 75



x_tensor = torch.Tensor(x_sample).cuda()  # unsqueeze gives a 1, batch_size dimension
y_tensor = torch.Tensor(y_sample).cuda()

torch_dataset = Data.TensorDataset(x_tensor, y_tensor)
    
loader = Data.DataLoader(
    dataset =torch_dataset,
    batch_size= BATCH_SIZE,
    shuffle = False,
    )

xtest_tensor = torch.Tensor(x_test).cuda()
ytest_tensor = torch.Tensor(y_test).cuda()

torch_testdataset = Data.TensorDataset(xtest_tensor, ytest_tensor)
    
testloader = Data.DataLoader(
    dataset =torch_testdataset,
    batch_size= BATCH_SIZE,
    shuffle = False,
    )


# decide on hyperparameters
input_size = feature_num
output_size = 1
hidden_dim =  20  # can be adjusted
#n_layers = 4


### instantiate an RNN
net = Net(input_size, hidden_dim, output_size)
net.cuda()
print(net)

### loss function and optimizer
# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
#lr = 0.0005
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)

hidden = None
TotalLoss = 0
Testloss = 0
global LOSS
global TestLOSS
global Record
global Count5
global Count10
global Count15
global Count20
global Count30

LOSS = []
TestLOSS = []
Record =[]
Count5 = []
Count10 =[]
Count15 =[]
Count20 =[]
Count30 =[]



for epoch in range(epoches):
    i =0
    for batch_x, batch_y in loader:
        i = i + 1
                    # outputs from the rnn
        prediction = net(batch_x)      # hidden for rnn no fc

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        #hidden = hidden.data
        # zero gradients
        optimizer.zero_grad()
        # calculate the loss
        loss = criterion(prediction, batch_y)

        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if i % print_every == 0:
            print('batch_i is: ',i)
            print('Loss: ', loss.item())
        TotalLoss = loss.item() + TotalLoss
    TotalLoss = TotalLoss/i
    scheduler.step()
    print('The average loss for epoche # ',epoch,' is ',TotalLoss)
    print("Learning Rate of %d epoch is：%f" % (epoch, optimizer.param_groups[0]['lr']))
    LOSS.append(TotalLoss)
    TotalLoss = 0
    Y_test =np.zeros((len(y_test)))
    j = 0
    for batch_x_test, batch_y_test in testloader:
        j =j + 1
        
        # test the data by the trained model
        test_out = net(batch_x_test)
        y_predict = test_out.cpu().data.numpy().flatten()
        loss = criterion(test_out, batch_y_test)
        Testloss = loss.item() + Testloss
        Y_test[(j-1)*BATCH_SIZE:j*BATCH_SIZE] = y_predict    
    Testloss = Testloss/j
    print('The average test loss for epoche # ',epoch,' is ',Testloss)
    TestLOSS.append(Testloss)
    Testloss =0
    
    Error = 0
    count5 = 0
    count10 = 0
    count15 = 0
    count20 = 0
    count30 = 0
    Total_Average_Error = 0
    for i in range(len(y_test)):
        error_percentage = np.abs((Y_test[i]-y_test[i])/y_test[i])    
        Error =Error + error_percentage
        if 0< error_percentage <= 0.05:
            count5 = count5 + 1
        elif 0.05 < error_percentage <= 0.1:
            count10 = count10 + 1     
        elif 0.1 < error_percentage <= 0.15:
            count15 = count15 + 1    
        elif 0.15 < error_percentage <= 0.20:
            count20 = count20 + 1
        else:
            count30 = count30 +1
                
        #print('The average error for validation dataset #:', i, 'between validation data set and predicted value is' ,Error)
        Total_Average_Error = Total_Average_Error + Error
        Error = 0    
    Record.append(Total_Average_Error/len(y_test))
    Count5.append(count5/len(y_test))
    Count10.append(count10/len(y_test))
    Count15.append(count15/len(y_test))
    Count20.append(count20/len(y_test))
    Count30.append(count30/len(y_test))
    print('The total average error between validation data set and predicted value is ',Total_Average_Error/len(y_test))
    print('The total number of predictions error 0-5% is',count5, 'the ratio is: ',count5/len(y_test))
    print('The total number of predictions error 5-10% is',count10, 'the ratio is: ',count10/len(y_test))
    print('The total number of predictions error 10-15% is',count15, 'the ratio is: ',count15/len(y_test))
    print('The total number of predictions error 15-20% is',count20, 'the ratio is: ',count20/len(y_test))
    print('The total number of predictions error over 20% is',count30, 'the ratio is: ',count30/len(y_test))
    
torch.save(net, 'trained_DNN.pt')



### figure plot setup ###
Epoches = np.linspace(1,epoches,epoches)

zipped_lists = zip(Count5, Count10)

Below10 = [x + y for (x, y) in zipped_lists]

zipped_lists15 = zip(Below10, Count15)

Below15 = [x + y for (x, y) in zipped_lists15]

zipped_lists20 = zip(Below15, Count20)

Below20 = [x + y for (x, y) in zipped_lists20]


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(Epoches, LOSS, 'r', label = 'Trainging Loss')
axs[0, 0].plot(Epoches, TestLOSS, 'b', label = 'Validation Loss')
axs[0, 0].legend()
axs[0, 0].set_title("Training Loss vs Validation Loss")
axs[1, 0].plot(Epoches, Below10)
axs[1, 0].set_title("Accuracy of 10% error tolerance")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(Epoches, Below15)
axs[0, 1].set_title("Accuracy of 15% error tolerance")
axs[1, 1].plot(Epoches, Below20)
axs[1, 1].set_title("Accuracy of 20% error tolerance")
fig.suptitle('Result of DNN Training of {} epoches'.format(epoches))
fig.tight_layout()
        
#fig.savefig("vector.svg")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

