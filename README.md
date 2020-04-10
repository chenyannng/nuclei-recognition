## Nuclei Segmentation
A Convolutional Neural Network (<a href="https://arxiv.org/abs/1505.04597">U-Net</a>) for nuclei recogniton problem using the dataset from <a href="https://www.kaggle.com/c/data-science-bowl-2018">Data Science Bowl 2018</a>.

1. Exploratory work on the image properties, image transformation: 
<br/>Use image_explore.ipynb
2. Build and train the CNN model: 
<br/>Run prepare_data.py : load, preprocess and save image data matrix
<br/>Run unet.py: main code for unet model
