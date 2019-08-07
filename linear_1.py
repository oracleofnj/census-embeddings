import numpy as np
import torch
from torch import nn, optim
from torch.utils import data


class LinearRegressionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class EmbeddingModule(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, x):
        out = self.embeddings(x)
        return out


class LinearEmbeddingModule(nn.Module):
    def __init__(self, embeddings, output_dim):
        super().__init__()
        self.embeddings = embeddings
        self.linear = LinearRegressionModule(
            embeddings.embedding_dim,
            output_dim
        )

    def forward(self, x):
        embeddings = self.embeddings(x)
        out = self.linear(embeddings).view(-1)
        return out


class LinearDataset(data.Dataset):
    def __init__(self, x, alpha, beta):
        super().__init__()
        self.x = x
        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        y = self.alpha + self.beta * self.x[index]
        return torch.tensor(index), torch.tensor(y)


np.random.seed(42)
x = np.random.rand(50)
y_1 = LinearDataset(x, 2, 0.5)
y_2 = LinearDataset(x, -1, 0)
y_gen_1 = data.DataLoader(y_1, batch_size=10)
y_gen_2 = data.DataLoader(y_2, batch_size=10)

embeddings = EmbeddingModule(50, 1)
model_1 = LinearEmbeddingModule(embeddings, 1)
model_2 = LinearEmbeddingModule(embeddings, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD([
    {'params': model_1.linear.parameters()},
    {'params': model_2.linear.parameters()},
    {'params': model_1.embeddings.parameters(), 'lr': 0.5},
], lr=0.01)

for epoch in range(200):
    for indices, labels in y_gen_1:
        optimizer.zero_grad()
        outputs_1 = model_1.forward(indices)
        loss_1 = criterion(outputs_1, labels)
        loss_1.backward()
        optimizer.step()

    for indices, labels in y_gen_2:
        optimizer.zero_grad()
        outputs_2 = model_2.forward(indices)
        loss_2 = criterion(outputs_2, labels)
        loss_2.backward()
        optimizer.step()

    if ((epoch + 1) % 10) == 0:
        print('epoch {0}, loss 1 {1}, loss 2 {2}'.format(
            epoch,
            loss_1,
            loss_2
        ))

print(model_1.linear.state_dict())
print(model_2.linear.state_dict())
print(np.corrcoef(x, embeddings.embeddings.weight.data, rowvar=False))
