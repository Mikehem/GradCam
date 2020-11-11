# Saliency Mapping for CV model interpretability

While deep learning has facilitated unprecedented accuracy in image classification, object detection, and image segmentation, one of their biggest problems is model interpretability, a core component in model understanding and model debugging.
In practice, deep learning models are treated as “black box” methods, and many times we have no reasonable idea as to.

  - Where the network is “looking” in the input image
  - Which series of neurons activated in the forward-pass during inference/prediction
  - How the network arrived at its final output?  
That raises an interesting question — how can you trust the decisions of a model if you cannot properly validate how it arrived there? 

**Saliency maps** are heat maps that are intended to provide insight into what aspects of an input image a convolutional neural network is using to make a prediction.

In this application we are going to look at 2 main categories of methods:
1. GradCam based methods
2. Integrated Gradients.

# GradCam!
There are 3 main core methods involved today
  - Gradients 
  - DeconvNets
  - Guided Backpropagation 

All these methods produce visualizations intended to show which inputs a neural network is using to make a particular prediction. They have been used for weakly supervised object localization (because the approximate position of the object is highlighted) and for gaining insight into a network’s misclassifications.  

<img src="https://da2so.github.io/assets/post_img/2020-08-10-GradCAM/2.png" alt="GradCam Architecture" width="800" height="400"> 

# Integrated Gradient!
Integrated gradients is a method originally proposed in Sundararajan et al., “Axiomatic Attribution for Deep Networks” that aims to attribute an importance value to each input feature of a machine learning model based on the gradients of the model output with respect to the input. In particular, integrated gradients defines an attribution value for each feature by considering the integral of the gradients taken along a straight path from a baseline instance x′ to the input instance x.  
[For more information click this link](https://keras.io/examples/vision/integrated_gradients/)

### Reference
•	[Gradients](https://arxiv.org/abs/1312.6034): Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv 2013 Cited by 1,720  
•	[DeconvNets](https://arxiv.org/abs/1311.2901): Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. ECCV 2014 Cited by 7,131  
•	[Guided Backpropagation](https://arxiv.org/abs/1412.6806): Springenberg JT, Dosovitskiy A, Brox T, Riedmiller M. Striving for simplicity: The all convolutional net. arXiv 2014 Cited by 1,504  
•	[Grad Cam](https://arxiv.org/abs/1610.02391):  Visual Explanations from Deep Networks via Gradient-based Localization.  
•	[Grad Cam++](https://arxiv.org/abs/1710.11063): Improved Visual Explanations forDeep Convolutional Networks  


# Running the Program
The App is built using Streamlit and requires tensorflow 2+ , alibi, pillow, matplotlib, pandas, numpy, opencv2

- recommended to use a virtuale nvironment such as conda.  
- run the app from the src directory : *streamlit run SaliencyMapping.py*  

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
