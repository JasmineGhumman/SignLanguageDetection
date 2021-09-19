# Sign Language Recognition Using Deep Neural Networks

A classic Deep Neural Network
architecture is used to recognize the hand signs to
solve a problem of American Sign Language
Recognition and classification. This project is based
on a state of the art concept of deep residual
learning for image recognition.

## Resnet50 based model
The approach used as part of this project makes use of the classic Resnet50 model. Resnet stands for residual
networks and it makes use of residual training.

### Residual Learning
Deep convolutional neural networks have led to a
series of breakthroughs for image classification.
Many other visual recognition tasks have also
greatly benefited from very deep models. So, over
the years there is a trend to go deeper, to solve
more complex tasks and to also increase /improve
the classification/recognition accuracy. But, as we
go deeper; the training of neural network becomes
difficult and also the accuracy starts saturating and
then degrades also. Residual Learning tries to solve
both these problems.

### Transfer Learning 

Any machine learning task involves a basic process
of feeding the data into the a neural network and let
the model learning by updating the weights of the
neurons in the neural network.
But in this project, instead of training our deep
neural network from scratch, the technique of
Transfer learning has been employed. This
technique is more relevant as to how our brain
performs the task of recognising the image. Rather
than doing the effort of learning the basic features
like edges and lines from scratch, transfer learning
allows us to use the weights of a model that has
already been trained on a large number of
images.Transfer learning allows us to train deep
networks using significantly less data then we
would need if we had to train from scratch. With
transfer learning, we are in effect transferring the
“knowledge” that a model has learned from a
previous task, to our current one. The idea is that
the two tasks are not totally disjoint, and as such
we can leverage whatever network parameters that
model has learned through its extensive training,
without having to do that training ourselves.

## Model architecture

ResNet50 model: In order to fine tune the ResNet50 for the purpose of Sign recognition, the weight of the
earlier layers of the ResNet50 model are frozen,
such that they are not trained when the input
images are fed into the model used.This has been
done, because the CNN and max pool layers are
already trained to identify the low-level features
like edges, lines, boundaries shades etc. However,
for this model to recognize the images fed for the
recognition of the Signs,
Fully Connected Layers have been appended to the
ResNet50 default model. The model is then trained
such that the weights of these fully connected
hidden layers are able to classify the hand sign.
ReLU activation function is used for all the
convolutional layers and fully connected layers
apart from the last layer where a softmax function
has been used. Cross-entropy has been used to
calculate the loss in each epoch and Adam
optimizer has been used to update the weights of
each hidden layer. To train the model on the data
set is divided the data in to batch of 32 rows to
make the learning process faster and efficient given
the amount of the data involved. The whole data is
trained for 20 epochs.

The data has been trained primarily on Google
Colab, the system with specs: 1x Tesla K80 ,
compute 3.7, having 2496 CUDA cores , 12GB
GDDR5 VRAM. Average time taken by each
epoch in ideal conditions was 132 seconds which
was 3 times less than the time taken when the
model was trained using the CPU only.

Link to the dataset - https://github.com/loicmarie/sign-language-alphabet-recognizer/tree/master/dataset

## Result

![ASL Result](/images/Result.png)