# Training I3D models on the Chimp&See dataset

Competition entry (5th place) for the DrivenData Pri-matrix Factoriation competition. Based on Inflated 3D ConvNet (I3D) models

Find out more about the competition [here](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/)


### Main dependencies:
tensorflow, ffmpeg, sk-video, tqdm, dm-sonnet
For example, if you want to train on AWS you can choose as AMI `Deep Learning AMI with Source Code (CUDA 8, Ubuntu)` or `Deep Learning AMI with Source Code (CUDA 9, Ubuntu)` and run the following commands to install remaining packages:
`$ sudo apt-get install ffmpeg`
`$ sudo pip3 install sk-video`
`$ sudo pip3 install tqdm`
`$ sudo pip3 install dm-sonnet`
`$ sudo ldconfig /usr/local/cuda/lib64  #Resolve cuda import error`

### Instructions:
- Download raw data, submission format and training set labels from [Drivendata](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/data/)
- Convert videos from raw data to fixed resolution of 224x224. You can use the `convert\_videos` function in `primatrix\_dataset\_utils.py`
- Set hyperparameters and run `train.ipynb` to train the model