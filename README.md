
# Food-State-Classification

## 1) Project Goal:
In this project we will design a deep convolutional neural network to classify an image of a cooking object to one of its states. For example given an image of a “sliced tomato” or “sliced bread”, the network should give as output “sliced”.

## 2) Project Dataset:
The dataset contains 17 cooking objects (chicken/turkey, beef/pork, tomato, onion, bread, pepper, cheese, strawberry, ...) with 11 different states (whole, julienne, sliced, chopped, grated, paste, floured, peeled, juice, mixed, other). This dataset contains 9309 images. The dataset can be downloaded from [Here](https://drive.google.com/file/d/1HU0Z3X3OltW8oUlW_Kkqsz_6kA_ma2tX/view)

## 3) Used techniques
 - data augmentation
 -  different drop rates 
 - batch normalization 
 -  use Inception module 
 -  use residual connections 
 
 ## 4) Using
all you need to put the images you want to classify into test/anonymous file and run `` python app.py`` then you will find the predictions in `` predictions.json``
 
 ## 4) Results
 ### Loss
 ![enter image description here](https://github.com/ahmedbasemdev/Food-State-Classification/blob/main/Images/loss.png?raw=true)
 ### Accuracy
 ![enter image description here](https://github.com/ahmedbasemdev/Food-State-Classification/blob/main/Images/accuracy.png?raw=true)
 
 

