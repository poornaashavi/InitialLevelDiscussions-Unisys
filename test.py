import os
import re
import nibabel as nib
import pandas as pd
import numpy as np
import torch
from torch import nn
from fastai.vision import *
from tqdm.notebook import tqdm
# import torchio as tio
# from torchio import AFFINE, DATA
from torchvision import datasets, transforms, models
from torch import optim
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndi
import shutil

import pennylane as qml
from pennylane import numpy as np

torch.manual_seed(42)
np.random.seed(42)


import time
n_qubits = 5                # Number of qubits
step = 0.000004               # Learning rate
batch_size = 8              # Number of samples for each training step
num_epochs = 25              # Number of training epochs
q_depth = 1                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time() 


dev = qml.device("lightning.qubit", wires=n_qubits)

train_path = "/Users/poornaash/Desktop/h/train"
test_path = "/Users/poornaash/Desktop/h/test"

train_csv = "/Users/poornaash/Desktop/h/train_labels.csv"
folders = os.listdir(train_path)
print(len(folders))

train_data = pd.read_csv(train_csv)
train_data.head()


print(train_data.query("id == 'afe011cd1711c4b0586a41e0cbe81eaeb8337d4a'")['label'].values[0])

train_data_numpy = train_data.values
print(train_data_numpy[0:5])


from sklearn.model_selection import train_test_split
data_train, data_val= train_test_split(train_data_numpy, test_size = 0.25, random_state=42,stratify=train_data_numpy[:,1])


import os
f = open("./missing.txt", "w")
root_dir = train_path
for i in tqdm(range(0,int(train_data_numpy.shape[0]))):
  filename = str(train_data_numpy[i][0])
  img_path = root_dir +'/'+ filename + '.tif'
  if(os.path.exists(img_path) == False):
    f.write(str(train_data_numpy[i][0])+"\n")

f.close()

files = open("./missing.txt", "r") #opens the file in read mode
missing_img = files.read().splitlines() #puts the file into an array
print(missing_img[0:5])
print(len(missing_img))
files.close()

print(len(train_data_numpy))
print(len(data_train))
print(len(data_val))


from PIL import Image
import matplotlib.pyplot as plt
img_path = train_path +'/' + '0544c8a53f81a2ec0b293b8caa0d97092fbb0615' + '.tif'
# brain_vol = nib.load(img_path)
# image_array = []
# brain_vol_data = brain_vol.get_fdata()
# fig, axs = plt.subplots(10, 10, figsize=[10, 10])

# for idx, img in enumerate(range(100)):
#     axs.flat[idx].imshow(ndi.rotate(brain_vol_data[img, :, :], 90), cmap='gray')
#     axs.flat[idx].axis('off')
image_array = Image.open(img_path)
numpy_array = np.asarray(image_array)
print(numpy_array.shape)
plt.imshow(image_array)
plt.show()


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image 
import skimage.transform as skTrans
class CancerClass(Dataset):
  def __init__(self,csv_data,root_dir,transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.data = csv_data

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    filename = str(self.data[idx][0])
    img_path = self.root_dir +'/'+ filename + '.tif'
   
    image_array = Image.open(img_path)
    image_array1 = image_array
    
    del image_array
    
    label = int(self.data[idx][1])
    
    if self.transform:
      tensor_image = self.transform(image_array1)

    return tensor_image,label

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])





@qml.qnode(dev, interface="torch",diff_method="adjoint")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # # # Sequence of trainable variational layers
    for k in range(q_depth):
      qml.BasicEntanglerLayers(weights=q_weights, wires=range(n_qubits))
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.rand(q_depth * n_qubits)) # 2 for QAOA //rand
        self.post_net = nn.Linear(n_qubits, 2)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = torch.tensor(quantum_net(elem, self.q_params), dtype=torch.float32).unsqueeze(0)

            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)



model_hybrid = models.resnet18(pretrained=True)
model_hybrid.fc = DressedQuantumNet()
model_hybrid = model_hybrid.to(device)



model_hybrid.apply(init_weights)




batch_size = 16
train_set = CancerClass(data_train,train_path,transforms.Compose(
    [transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

val_set = CancerClass(data_val, train_path, transforms.Compose([transforms.ToTensor()]))
val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size, shuffle=False)


import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model_hybrid.parameters(),lr=1E-5)
exp_lr_scheduler = lr_scheduler.StepLR(
    opt, step_size=10, gamma=gamma_lr_scheduler
)




def evaluation(dataloader,model):
  with torch.no_grad():
    total,correct = 0,0
    for data in dataloader:
      inputs,labels = data
      inputs,labels = inputs.to(device),labels.to(device)
      outputs = model(inputs)
      _,pred = torch.max(outputs.data,1) 
     # pred = pred + 1
      total += labels.size(0)
      correct += (pred==labels).sum().item()
    
    return 100*correct/total




loss_arr = []
epoch_loss_arr = []
epoch_acc = []
import copy
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    print("Training started:")
    for epoch in range(num_epochs):
      a = 1
      model.train()
      for i, data in enumerate(train_loader,0):
        a = a+1
        print(a)
        inputs, labels = data
        batch_size_ = len(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_arr.append(loss.item())

      epoch_loss_arr.append(loss.item())
      epoch_acc.append(evaluation(train_loader,model))
      print(model.fc.q_params)
      print(
                "Epoch: {}/{} Train Loss: {:.4f} Train Acc: {:.4f} Val Acc: {:.4f} ".format(
                    epoch,
                    num_epochs,
                    epoch_loss_arr[-1],
                    epoch_acc[-1],
                    evaluation(val_loader,model)
                )
            )
      PATH = '/content/' + 'model_' + str(epoch) + '.pt'
      torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss.item(),
            }, PATH)
      #best_model_wts = copy.deepcopy(model.state_dict())
      # Update learning rate
      scheduler.step()
      #model.load_state_dict(best_model_wts)
       
    # Print final results
    return model



model_hybrid = train_model(
    model_hybrid, loss_fn, opt, exp_lr_scheduler, num_epochs=25
)

plt.plot(epoch_loss_arr)
plt.show()
