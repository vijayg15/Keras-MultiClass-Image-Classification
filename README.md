# Multi class Weather Classification
<p align="justify"> Time and again unfortunate accidents due to inclement weather conditions across the globe have surfaced. Ship collision, train derailment, plane crash and car accidents are some of the tragic incidents that have been a part of the headlines in recent times. This grave problem of safety and security in adverse conditions has drawn the attention of the society and numerous studies have been done in past to expose the vulnerability of functioning of transportation services due to weather conditions. 
In past, weather-controlled driving speeds and behaviour have been proposed. With the advancement in technology and emergence of a new field, intelligent transportation, automated determination of weather condition has become more relevant. Present systems either rely on series of expensive sensors or human assistance to identify the weather conditions. Researchers, in recent era have looked at various economical solutions. They have channelled their research in a direction where computer vision techniques have been used to classify the weather condition using a single image. The task of assessing the weather condition from a single image is a straightforward and easy task for humans. Nevertheless, this task has a higher difficulty level for an autonomous system and designing a decent classifier of weather that receives single images as an input would represent an important achievement. 


<p align="justify"> The aim of this capstone project is to build a convolutional neural network that classifies different weather conditions while working reasonably well under constraints of computation. The work described in this report translates to two contributions to the field of weather classification. The first one is exploring the use of data augmentation technique, considering different Convolutional Neural Network (CNN) architectures for the feature extraction process when classifying outdoor scenes in a multi-class setting using general-purpose images. The second contribution is the creation of a new, open source dataset, consisting of images collected online that depict scenes of five weather conditions.

<p align="justify"> Transfer Learning with ResNet and VGG neural network architecture on multi-class weather classification problem with dataset collected online containing Creative Commons license retrieved from Flickr, Unsplash and Pexels. The final model (<b>ResNet101</b>) on test-data yields a <b>log-loss</b> of <b><i>0.080738</i></b> with valid accuracy of <b>96.67</b>%.

<div align="center">
    <img height="350" width="400" src="https://github.com/vijayg15/Multi-class-Weather-Classification/blob/master/figures_and_plots/cm_wo_norm.png">
</div>

The final report with model visualizations and validation plots is [here](https://github.com/vijayg15/Multi-class-Weather-Classification/blob/master/wp_project_report.pdf).


# Dependencies :
- Python 3.5
- keras `conda install keras`
- tensorflow `conda install tensorflow`

Note that keras and tensorflow have their own dependencies. I recommend using [Anaconda](https://www.anaconda.com/) for handlinng the packages.

# Dataset :
The Weather dataset can be downloaded from [Kaggle](https://www.kaggle.com/vijaygiitk/multiclass-weather-dataset) (that I've uploaded there).
<p align="center">
    <img height="108" width="90" src="https://github.com/vijayg15/Multi-class-Weather-Classification/blob/master/figures_and_plots/fol.jpg">
</p>

# Usage : 
- Use wp_EDA_preprocessing Notebook to split training and validation data into respective folders.
<p align="center">
    <img height="196" width="96" src="https://github.com/vijayg15/Multi-class-Weather-Classification/blob/master/figures_and_plots/fol_train_val.jpg">
</p>
        
- Read the [wp_project_report](https://github.com/vijayg15/Multi-class-Weather-Classification/blob/master/wp_project_report.pdf) report for understanding the approaches taken in solving this problem
- Notebooks are under the [Notebooks folder](https://github.com/vijayg15/Multi-class-Weather-Classification/tree/master/Notebooks)  and scripts are under the [scripts folder](https://github.com/vijayg15/Multi-class-Weather-Classification/tree/master/scripts).
- All the model diagrams and performance charts are provided.
- The [model weights](https://github.com/vijayg15/Multi-class-Weather-Classification/releases) can be downloaded [here](https://github.com/vijayg15/Multi-class-Weather-Classification/releases) to predict the weather condition.
