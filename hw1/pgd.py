import torch
import torch.nn as nn
import tqdm


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 1 # target class

epsReal = 1.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros
iter_eps = 1e-1

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

x_now = x.clone().detach().requires_grad_()

for _ in tqdm.trange(100000):

    # compute gradient
    # note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
    L = nn.CrossEntropyLoss()
    loss = L(N(x_now), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
    loss.backward()

    adv_x = x - torch.clamp(x - (x_now - iter_eps * x_now.grad.sign()), -eps, eps)

    new_class = N(adv_x).argmax(dim=1).item()

    if new_class == t:
        break

    x_now = adv_x.clone().detach().requires_grad_()

print("Original Input: ", x)
print("Adversarial Example: ", adv_x)
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print("Norm: ", torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)