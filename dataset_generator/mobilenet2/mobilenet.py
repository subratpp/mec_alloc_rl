#!/usr/bin/env python
# coding: utf-8

# # MobileNet V2 Pytorch

# In[1]:

import sys

import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.eval()


# In[2]:


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

#number of input images
filename = "dog.jpg"
# Load image and create tensor

num_of_users = int( sys.argv[1] ) #read the user argument
tensor_list = [] #list of image tensors
#preprocess images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#create list of image tensors according to number of users
for i in range(num_of_users):
    input_image = Image.open(filename)
    tensor_list.append(preprocess(input_image))


# In[3]:


#create batch of images for parallel processing
input_tensor = torch.stack(tensor_list)


# In[4]:



# In[5]:


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_tensor = input_tensor.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_tensor)  


# In[6]:


# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output, dim=1)
# print(probabilities)


# In[7]:


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 1)
# for i in range(top5_prob.size(0)):
#     print(categories[ top5_catid[i].tolist()[0] ])

print(">>>>>MobileNetv2 Processed! Done!")
# In[ ]:




