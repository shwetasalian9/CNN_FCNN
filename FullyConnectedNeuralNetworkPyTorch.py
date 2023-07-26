import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import csv

def simple_gradient():
	# print the gradient of 2x^2 + 5x
	x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
	z = 2 * (x * x) + 5 * x
	# run the backpropagation
	z.backward(torch.ones(2, 2))
	print(x.grad)


def write_csv(y_list):
    solution_rows = [('id', 'category')] + [(i, y) for (i, y) in enumerate(y_list)]
    with open('/content/drive/My Drive/Startingcode/FCNN.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)	



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(3*32*32, 500)
		self.fc2 = nn.Linear(500, 400)
		self.fc3 = nn.Linear(400, 300)
		self.fc4 = nn.Linear(300, 200)
		self.fc5 = nn.Linear(200, 150)
		self.fc6 = nn.Linear(150, 10)



	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc6(x)
		return F.log_softmax(x)


  # create a two hidden layers fully connected network
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=2)
print(train_loader)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
print(net)

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()

# run the main training loop
for epoch in range(10):
	print(train_loader)
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		

		#print(data.shape)
		#print(target.shape)
		# resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
		data = data.view(-1, 3*32*32)
		optimizer.zero_grad()
		net_out = net(data)
		loss = criterion(net_out, target)
		loss.backward()
		optimizer.step()
		# if batch_idx % 10 == 0:
		# 	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		# 		epoch, batch_idx * len(data), len(train_loader.dataset),
		# 			   100. * batch_idx / len(train_loader), loss.data[0]))


#Validate
val_loss = 0
correct_val = 0

for data, target in val_loader:
	data, target = Variable(data, volatile=True), Variable(target)
	data = data.view(-1, 3*32*32)
	net_out = net(data)
	# sum up batch loss
	val_loss += criterion(net_out, target)
	pred = net_out.data.max(1)[1]  # get the index of the max log-probability
	
	correct_val += pred.eq(target.data).sum()

val_loss /= len(val_loader.dataset)
print('\nValidation set: Average loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
	val_loss, correct_val, len(val_loader.dataset),
	100. * correct_val / len(val_loader.dataset)))	

# run a test loop
test_loss = 0
correct = 0
pred= None
for data, target in test_loader:
	data, target = Variable(data, volatile=True), Variable(target)
	data = data.view(-1, 3*32*32)
	net_out = net(data)
	# sum up batch loss
	test_loss += criterion(net_out, target)
	pred = net_out.data.max(1)[1]  # get the index of the max log-probability
	
	correct += pred.eq(target.data).sum()

pred = pred.cpu().data.numpy()
pred = list(pred)
write_csv(pred)
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)\n'.format(
	test_loss, correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))



if __name__ == "__main__":
	run_opt = 2
	if run_opt == 1:
		simple_gradient()