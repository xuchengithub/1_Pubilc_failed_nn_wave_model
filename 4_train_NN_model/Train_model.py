# tensorboard --logdir=runs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
############## CODE-FOR-DATA ######################
import time
from dataset import data_set_for_train
from torch.utils.data import DataLoader

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

############## CODE ###############################
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# time
now_time = time.asctime(time.localtime(time.time()))
now_time = now_time.replace(" ", "_")
now_time = now_time.replace(":", "-")

# Hyper-parameters
input_size = 10050  # 75*134=1005
hidden_size = 500
num_hidden_layers = 4
num_classes = 1

num_epochs = 150

batch_size = 50
learning_rate = 0.0001
val_running_accuracy = 0
# load-data-from
train_data_address = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/all_train_data.npy"
var_data_address = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/data4/data_used_for_train_model_all.npy"

# change-data-format
dataset_train = data_set_for_train(train_data_address)
dataset_val = data_set_for_train(var_data_address)

train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    dataset_val, batch_size=batch_size, num_workers=4, pin_memory=True
)
print("load_data_finished")

############# TENSORBOARD ########################
examples = iter(test_loader)
example_data, example_targets = examples.next()
# images_list=list()
# for i in range(example_data.shape[0]):
#     images_list.append(example_data[i])
# img_grid = torchvision.utils.make_grid(images_list)
example_data = torch.unsqueeze(example_data, 1)
writer.add_images('test_data', example_data)
# writer.close()
# sys.exit()
###################################################

# Fully connected neural network with one hidden layer


model = NeuralNet(input_size, num_hidden_layers,
                  hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = F.binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# torch.optim.SGD
############## TENSORBOARD ########################
reset_example_data = torch.unsqueeze(example_data, 1)
writer.add_graph(model, reset_example_data.to(device))
# writer.close()
# sys.exit()
###################################################

# Train the model
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
How_much_steps_to_save_data = n_total_steps
for epoch in range(num_epochs):
    num_of_data = 0
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        print(f"time:::::::::::::::::::::::{i}")
        images = torch.unsqueeze(images, 1).to(device)
        # images = images.reshape(-1, 28*28)
        labels = torch.unsqueeze(labels, 1)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        out_result = outputs.data > 0.5
        out_labels = labels == 1
        # _, predicted = torch.max(outputs.data, 1)
        running_correct += (out_result == out_labels).sum().item()
        num_of_data += out_result.size(0)
        if (i+1) % How_much_steps_to_save_data == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            ############## TENSORBOARD ########################

        if (i+1) == n_total_steps:
            writer.add_scalar('training loss', running_loss /
                              num_of_data, (epoch+1) * n_total_steps)
            running_accuracy = running_correct / num_of_data
            writer.add_scalar('training accuracy', running_accuracy,
                              (epoch+1) * n_total_steps + i)
            print(
                f'Epoch [{epoch+1}/{num_epochs}],training_accuracy: {round(running_accuracy, 4) }')
            running_correct = 0
            running_loss = 0.0

            with torch.no_grad():
                val_num_of_data = 0
                val_running_correct = 0
                val_n_total_steps = len(test_loader)
                for val_i, (val_images, val_labels) in enumerate(test_loader):
                    val_images = torch.unsqueeze(val_images, 1).to(device)
                    val_labels = torch.unsqueeze(val_labels, 1)
                    val_labels = torch.tensor(
                        val_labels, dtype=torch.float32).to(device)
                    val_outputs = model(val_images)
                    # max returns (value ,index)
                    val_out_result = val_outputs.data > 0.5
                    val_out_labels = val_labels == 1
                    # _, predicted = torch.max(outputs.data, 1)
                    val_running_correct += (val_out_result ==
                                            val_out_labels).sum().item()
                    val_num_of_data += val_out_result.size(0)

                    if (val_i+1) == val_n_total_steps:
                        # print(f'Step [{val_i+1}/{val_n_total_steps}]')
                        ############## TENSORBOARD ########################
                        val_running_accuracy = val_running_correct / val_num_of_data
                        writer.add_scalar(
                            'val_accuracy', val_running_accuracy, (epoch+1) * n_total_steps + val_i)
                        print(
                            f'val_accuracy: {round(val_running_accuracy, 4) }')

###################################################
torch.save(
    model, f"model_epochs_{num_epochs}_lr_{learning_rate}_acc_{val_running_accuracy}_time_{now_time}.pt")
writer.close()
sys.exit()
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# class_labels = []
# class_preds = []

# class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

# class_preds.append(class_probs_batch)
# class_labels.append(predicted)

# # 10000, 10, and 10000, 1
# # stack concatenates tensors along a new dimension
# # cat concatenates tensors in the given dimension
# class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
# class_labels = torch.cat(class_labels)

# acc = 100.0 * n_correct / n_samples
# print(f'Accuracy of the network on the 10000 test images: {acc} %')

############## TENSORBOARD ########################
# classes = range(10)
# for i in classes:
#     labels_i = class_labels == i
#     preds_i = class_preds[:, i]
#     writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
#     writer.close()
###################################################
