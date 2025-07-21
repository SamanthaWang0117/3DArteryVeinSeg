# main.py
import torch
import torch.nn as nn
import numpy as np
from models.3DVNet import VNET


if __name__ == '__main__':
    dim = 2
    net = VNET(cross_hair=True, dim=dim, nlabels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    
    N_batch = 10
    n_channels = 1
    image_size = 64
    n_labels = 2

    X = torch.randn((N_batch, n_channels) + (image_size,) * dim, dtype=torch.float32).to(device)
    Y_np = np.random.randint(n_labels, size=(N_batch,) + (image_size,) * dim)
    Y_one_hot_np = np.eye(n_labels)[Y_np]
    if dim == 3:
        Y_one_hot_np = np.transpose(Y_one_hot_np, (0, 4, 1, 2, 3))
    else:
        Y_one_hot_np = np.transpose(Y_one_hot_np, (0, 3, 1, 2))
    Y = torch.from_numpy(Y_one_hot_np).float().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print('Testing VNET Network in PyTorch')
    print('Data Information => ', 'volume size:', X.shape, ' labels:', Y.shape)

    num_epochs = 10
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        outputs = net(X)
        loss = loss_fn(outputs, Y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("PyTorch VNET testing complete.")