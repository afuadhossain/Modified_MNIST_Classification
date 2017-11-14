COMP 551 Project 3 - Image Analysis

Objective: Devise a machine learning algorithm to automatically compute mathematical functions from images.

Note: Please retrieve the relevant .csv files from https://www.kaggle.com/c/comp551-modified-mnist/data and place them in the data directory.



________________________HOW TO RUN____________________________
1)Pickle the data

   a) Retrieve the relevant .csv files from https://www.kaggle.com/c/comp551-modified-mnist/data and place them in the data directory.
   
   b) Write the following in a command line prompt 
   
         cd ~/ARBITRARY_PATH/comp551-project3-master
         
         python preprocess/pickle_data.py
   
   
2) Run the CNN predictor
   
   a) make sure to run in an (virtual) environment that supports tensorflow. Information regarding this is found at https://www.tensorflow.org/install/
   
   b) if using a virtual environment, activate it. 
   
   b) The model number of the CNN must be changed every time it is re-trained. 
   line 117:  model_fn=cnn_model_fn, model_dir="/tmp/comp551_convnet_model212")
      - change "/tmp/comp551_convnet_model212" to something else such as "/tmp/comp551_convnet_model213"
   
   c) Write the following in a command line prompt 
          
          cd ~/ARBITRARY_PATH/comp551-project3-master
          python Convolutional-Neural-Net/predictCNN.py



____________DESCRIPTION OF DIRECTORIES AND FILES_______________
Directory: preprocess

    File: pickle_data.py
          This script reads in the data in csv form, preprocesses it, and saves it as a .pkl file. These are larger, but are    in the numpy array form so they are faster to read in. It saves the pkl files in the 'data' directory.
          
    File: image_processing.py
          This contains the methods for preprocessing the images used in pickle_data.py
 
_______________________________________________________________
Directory: Convolutional-Neural-Net
    
    File: cnn.py
           This is a very basic CNN architecture. Was only used to learn how to set one up using tensor flow. Ran it for 500 steps and got like 12% accuracy using .25 cross validation
           
          
    File: cnn_alternate.py
          This is a more advanced CNN inspired by an architecture used to detect Chinese characters. This is what we hope to use to prediction     
