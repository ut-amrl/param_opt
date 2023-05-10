import torch
import csv


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input size is a 41 x 4 x 1 tensor
        # 2 convolutional layers then a densely connected layer to output to size 41
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 4 * 41, 41)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 41)
        x = self.fc1(x)
        return x


class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.data = self.read_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def read_data(self):
        # read a csv file
        # the first row is (length, n_samples, n_features, 1)
        # each row is the row major enumeration of a n_samples x n_features x 1 tensor followed by a n_samples x 1 tensor
        # stores the data in a list of tuples of tensors
        data = []
        with open("data.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            header = next(reader)
            n_samples = int(header[1])
            n_features = int(header[2])
            for row in reader:
                state = torch.tensor(
                    [float(x) for x in row[: n_samples * n_features]]
                ).view(1, n_samples, n_features)
                cost = torch.tensor(
                    [
                        float(x)
                        for x in row[
                            n_samples * n_features : n_samples * n_features + n_samples
                        ]
                    ]
                ).view(1, n_samples)
                data.append((state, cost))
        return data


net = Network().cuda()
data = Data()
train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss()

losses = []
for epoch in range(5):
    l = 0
    for X, y in train_loader:
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        output = net(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        l += loss.item()
    losses.append(l / len(train_loader))
    print("Epoch: %d, Loss: %.4f" % (epoch, l / len(train_loader)))

c = 0
for X, y in data:
    X, y = X.cuda(), y.cuda()
    gt = y.argmin().cpu().detach().item()
    output = net(X)
    pred = output.argmin().cpu().detach().item()
    if gt == pred:
        c += 1
    # elif gt == pred + 1 or gt == pred - 1:
    #     c += 0.5
print("Accuracy: %.4f" % (c / len(data)))
