import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


N = 100

def generate_data():
    indices = np.random.choice(4, N)
    features = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])

    selected_features = torch.from_numpy(features[indices]).float()
    generated_x = torch.cat([torch.randn(N,1), selected_features], dim=1)
    labels = torch.from_numpy((indices==0)|(indices==3)).float()

    return generated_x, labels

generated_x, labels = generate_data()

net_embed = nn.Sequential(
        nn.Linear(3, 256),
        nn.ReLU(),
        nn.Linear(256,64)
        )
net_predict = nn.Linear(64,1)

opt = optim.Adam(list(net_embed.parameters())+list(net_predict.parameters()))

for i in range(20):
    predictions = net_predict(net_embed(generated_x)).squeeze()
    loss = nn.functional.binary_cross_entropy_with_logits(predictions, labels)
    print("Loss:", loss.item())
    net_embed.zero_grad()
    net_predict.zero_grad()
    loss.backward()
    opt.step()
    


test_data, test_labels = generate_data()

final_predictions = torch.sigmoid(net_predict(net_embed(test_data)).squeeze())
#print(labels)
accuracy = torch.sum(torch.abs(final_predictions - test_labels) < 0.5).item() / N
print("Accuracy=",accuracy)

