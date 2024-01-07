import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim.lr_scheduler as lr_scheduler

NUM_TRAIN = 10000
NUM_VAL = 1000
NUM_TEST = 1000
epochs = 50
device = torch.device("cpu")
torch.manual_seed(42)

def get_data_simple_cancer(size):

    # Initial tumor size uniform between possible sizes
    X0 = torch.Tensor(size).uniform_(0, 1)

    # Applying therapy or not bernoulli of probability deppending on the size
    A0 = torch.bernoulli(X0)

    # We normalize the data (we do not care about saving mean and std to get real data back)
    #X0 = (X0 - torch.mean(X0))/torch.std(X0)

    # If we apply therapy, half tumor size (if A1 has effect on Y)
    Y1 = []
    for i, x in enumerate(X0):
        if(A0[i] == 0):
            Y1.append(x)
        else:
            Y1.append(x)

    # We create the dataset for the BalanceNet
    dataset = []
    for i in range(len(Y1)):
        dataset.append([torch.tensor((X0[i], A0[i])), torch.tensor((Y1[i], A0[i]))])
    
    return dataset

# Basic Gradient Reverse Layer with scale = 1
class GradientReverse(torch.autograd.Function):
    scale = 1
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
# Function to easily reverse the gradient
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

# Loss to avoid predicting A1 if balance = True
class BalancingLoss(nn.Module):
    def __init__(self, balance=True, ini_lambda_param = 0, gamma = 10):
        super(BalancingLoss, self).__init__()
        self.p = 0
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss()
        self.balance = balance
        self.ini_lambda_param = ini_lambda_param
        self.gamma = gamma

    def forward(self, inputs, targets):
        #Mean Square Error of Y
        Y_mse = self.MSE(inputs[0], targets[0])

        #Binary Cross Entropy of A1
        A_bce = self.BCE(inputs[1], targets[1])

        if(self.balance):
            #Schedule of lambda on the original Domain Adversarial Paper 
            if(self.ini_lambda_param == 0):
                lambda_param = 2/(1 + np.exp(-self.gamma*self.p)) - 1
                self.p += 1/(NUM_TRAIN * epochs)
            else:
                lambda_param = self.ini_lambda_param
        else:
            lambda_param = 0
        
        #Final loss
        res =  Y_mse + lambda_param * A_bce
        return res

# Neuran Network to predict A1 without backpropagatting into the representation
class ANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_elu_stack_A = nn.Sequential(
            nn.Linear(1, 48),
            nn.ELU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, phi):
        pred_a = self.linear_elu_stack_A(phi)
        return pred_a

# Neural Network described in the Iona Bica paper
class BalancerNet(nn.Module):
    def __init__(self, balance):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_elu_stack_phi = nn.Sequential(
            nn.Linear(1, 48),
            nn.ELU(),
            nn.Linear(48, 1),
        )
        self.linear_elu_stack_Y = nn.Sequential(
            nn.Linear(2, 48),
            nn.ELU(),
            nn.Linear(48, 1),
        )
        self.linear_elu_stack_A = nn.Sequential(
            nn.Linear(1, 48),
            nn.ELU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )
        self.balance = balance

    def forward(self, x):
        L0, A0 = x
        L0 = L0.reshape(1)
        A0 = A0.reshape(1)

        # Calculate representation
        phi = self.linear_elu_stack_phi(L0)

        # Adversarial training for predicting A_0
        if(self.balance):
            phi_rev = grad_reverse(phi)
            logit_a = self.linear_elu_stack_A(phi_rev)
        else:
            logit_a = torch.tensor(0.5).reshape(1)
            
        # Predict Y using representation and A_1
        pred_y = self.linear_elu_stack_Y(torch.cat((phi, A0)))

        return torch.cat((pred_y, logit_a, phi))

def train_loop(dataset, model, loss_fn, optimizer, balance):
    model.train()
    check_loss = BalancingLoss(False)
    y_loss = 0
    accuracy_prediction_a = 0
    for (X, y) in dataset:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred[:-1], y)
        y_loss += check_loss(pred[:-1], y).detach().numpy()

        # Compute if predicts A1
        if(((pred[1] > 0.5) and y[1] == 1) or ((pred[1] <= 0.5) and y[1] == 0)):
                accuracy_prediction_a += 1
    
        # Backpropagation
        loss.backward()

        #Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

        optimizer.step()
        optimizer.zero_grad()
    
    # Compute Root square mean error
    y_loss = np.sqrt(y_loss/len(dataset))

    if(balance):
        # Compute accuray of predicting A1
        accuracy_prediction_a = accuracy_prediction_a/len(dataset) * 100
    else:
        # If we do not balance, we are randomly predicting A1
        accuracy_prediction_a = torch.nan
    
    print(f"Train Error: \n RMSE loss: {y_loss:>8f} Accuracy prediction A: {accuracy_prediction_a}% \n")


def test_loop(dataset, model, balance):
    # Set the model to evaluation mode 
    model.eval()
    test_loss = 0.0
    loss = BalancingLoss(False)
    accuracy_prediction_a = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataset:
            Y, A1 = y
            pred = model(X)
            test_loss += loss(pred[:-1], y).item()
            if(((pred[1] > 0.5) and A1 == 1) or ((pred[1] <= 0.5) and A1 == 0)):
                    accuracy_prediction_a += 1

    if(not balance):
        accuracy_prediction_a = torch.nan
    test_loss = np.sqrt(test_loss/len(dataset))
    accuracy_prediction_a = accuracy_prediction_a/len(dataset) * 100
    print(f"Val Error: \n RMSE loss: {test_loss:>8f} Accuracy prediction A: {accuracy_prediction_a}% \n")
    
    return test_loss

#  External A1 predictor
def train_A(dataset, model, model_a, optimizer_a):
    loss_a = torch.nn.BCELoss()
    for X, y in dataset:
        Y, A0 = y
        with torch.no_grad():
            _, _, phi = model(X)
        pred = model_a(phi.detach().reshape(1))
        loss_res_a = loss_a(pred, A0.reshape(1))

        # Backpropagation
        loss_res_a.backward()
        optimizer_a.step()
        optimizer_a.zero_grad()

# Validate how good we predict A1
def test_A(dataset, model, model_a):
    model.eval()
    accuracy_prediction_a = 0

    with torch.no_grad():
        for X, y in dataset:
            _, A0 = y
            _, _, phi = model(X)
            A_pred = model_a(phi.reshape(1))
            if(((A_pred > 0.5) and A0 == 1) or ((A_pred <= 0.5) and A0 == 0)):
                    accuracy_prediction_a += 1
    accuracy_prediction_a = accuracy_prediction_a/len(dataset)
    print(f"Accuracy prediction A no balance: {accuracy_prediction_a * 100}% \n")
    return accuracy_prediction_a


dataset_train = get_data_simple_cancer(NUM_TRAIN)
dataset_val = get_data_simple_cancer(NUM_VAL)
dataset_test = get_data_simple_cancer(NUM_TEST)


def experiment(balance, warmup, A_predictor, ini_lambda = 0, gamma=10):
    epochs = 50
    loss_fn = BalancingLoss(balance, ini_lambda, gamma)
    model = BalancerNet(balance)
    model_a = ANet()

    # We use SDG with momentum 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=50)

    best_loss = torch.inf
    no_improve_counter = 0
    best_model_epoch = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataset_train, model, loss_fn, optimizer, balance)
        res_loss = test_loop(dataset_val, model, balance)
        if ((t >= warmup) and (res_loss < best_loss)):
            best_loss = res_loss
            no_improve_counter = 0
            best_model = copy.deepcopy(model)
            best_model_epoch = t
        elif(t >= warmup):
            no_improve_counter += 1
        if(no_improve_counter >= 4):
            break
        scheduler.step()
    
    print("TEST RESULTS: ")
    test_loop(dataset_test, best_model, balance)
    print("Best model epoch: ", best_model_epoch)

    if(A_predictor):
        print("Training A external predictor:")
        best_accuracy = 0
        no_improve_counter = 0
        epochs = 50
        warmup = 10
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_A(dataset_train, best_model, model_a, optimizer_a)
            accuracy = test_A(dataset_val, best_model, model_a)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improve_counter = 0
                best_model_a = copy.deepcopy(model_a)
            elif(t >= warmup):
                no_improve_counter += 1
            
            if(no_improve_counter >= 4):
                break
        print("Test A results:")
        test_A(dataset_test, best_model, best_model_a)

    print("Done!")



print("Experiment with balance lambda = 1")
learning_rate = 1e-4
experiment(True, 15, True, 1, 10)

print("Experiment with DA balance, gamma = 10:")
learning_rate = 1e-4
experiment(True, 30, True)

print("Experiment without balance:")
learning_rate = 1e-4
experiment(False, 0, True)
