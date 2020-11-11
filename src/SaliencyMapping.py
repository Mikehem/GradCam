from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#-----------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.resnet_v2 import decode_predictions as resnet_decode_predictions
import tensorflow as tf
from tensorflow.python.framework import ops
# Display
from PIL import Image
#-----------------------------------------
from alibi.explainers import IntegratedGradients
from alibi.datasets import fetch_imagenet
from alibi.utils.visualization import visualize_image_attr
#------------------------------------------
from saliency.gradcam import GradCAM, overlay_gradCAM
from saliency.guidedBackprop import GuidedBackprop, deprocess_image
from saliency.utils import preprocess, predict, predict_5, SAMPLE_DIR, array2bytes, DECODE, INV_MAP
from saliency.models import load_ResNet50PlusFC, load_VanilaResNet50, load_ResNet50, load_ResNet50V2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.set_option('deprecation.showPyplotGlobalUse', False)
# Input shape, defined by the model (model.input_shape)
H, W = 224, 224 

CUSTOM_CLASS_DECODE = {0:"Cat", 1:"Dog"}
CUSTOM_CLASS_INV_MAP = {"Cat": 0, "Dog": 1}

def load_image(x, preprocess=True):
    """Load and preprocess image."""
    # Resize image using opencv
    #x = np.resize(W,H)
    if preprocess:
        x = keras_image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = vgg16_preprocess_input(x)
    return x

def deprocess(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)
        
    with tf.GradientTape() as tape:
        x = -tf.math.reduce_mean(x)
        x /= (tf.math.reduce_std(x) + 1e-5) 
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 25
        if K.image_data_format() == 'th':
            x = x.transpose((1, 2, 0))
            
        x = np.clip(x, 0, 255).astype('uint8')
        
    return x

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def vgg16_build_model():
    """Function returning keras model instance.
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    return VGG16(include_top=True, weights='imagenet')

def vgg16_grad_cam(input_model, image_array, last_conv_layer_name, classifier_layer_names, cls=-1):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    model = input_model
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    # Second, we create a model that maps the activations of the last conv layer 
    # to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(image_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        if cls == -1:
            pred_index = tf.argmax(preds[0])
        else:
            pred_index = cls
        class_channel = preds[:, pred_index]
        #print(class_channel)
        
    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

def vgg16_guided_backprop(model, image_array, layer_name, CLASS_INDEX):
    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(layer_name).output, model.output])
    
    layer_dict = [layer for layer in grad_model.layers[1:] if hasattr(layer,"activation")]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu
    
    # Get the score for target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, CLASS_INDEX]
    
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Apply guided backpropagation
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    
    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    return cam

def vgg16_compute_predictions(model = None, img_path = None, top_n = 5):
    """ Compute saliency using all three approaches.
        - layer_name: layer to compute gradients;
        - cls: class number to localize (-1 for most probable class).
    """
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    top = vgg16_decode_predictions(predictions, top=top_n)[0]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    st.header('Model prediction')
    c_list, p1_list, p2_list = [] , [], []
    for c, p in zip(classes, top):
        c_list.append(c)
        p1_list.append(p[1])
        p2_list.append(p[2])
        #st.write('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    st.write(pd.DataFrame(zip(p1_list, c_list, p2_list), columns=['Class Name', 'Class ID', 'Probaility']))
    return classes
        
def compute_vgg16_saliency(model = None, img_path = None, layer_name='block5_conv3', classifier_layers = "predictions", cls=-1, 
                     visualize=True, sap1_list = None, save=False):
    """ Compute saliency using all three approaches.
        - layer_name: layer to compute gradients;
        - cls: class number to localize (-1 for most probable class).
    """
    preprocessed_input = load_image(img_path)
    predictions = model.predict(preprocessed_input) 
    cls = int(cls)
    top_1 = [cls]
    if cls == -1:
        cls = np.argmax(predictions)
        top_1 = np.argsort(predictions[0])[-1:][::-1]
    class_name = vgg16_decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    
    st.header("\nExplanation for '{}' ({})\n".format(class_name, top_1))
    
    # Grad Cam
    #gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gradcam = vgg16_grad_cam(input_model = model, image_array = preprocessed_input, 
                       last_conv_layer_name = layer_name, classifier_layer_names = classifier_layers,
                      cls=cls)
    
    # We rescale heatmap to a range 0-255
    gradcam_heatmap = np.uint8(255 * gradcam)
    
    # image
    preprocessed_img = preprocessed_input.reshape(224, 224, 3)
    
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[gradcam_heatmap]
    
    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((preprocessed_img.shape[1], preprocessed_img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + preprocessed_img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    

    # Guided Backpropogation
    #gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    gb_cam = vgg16_guided_backprop(model = model, image_array = preprocessed_input, 
                             layer_name = layer_name, CLASS_INDEX = top_1[0])
    
    # Visualise 
    # We rescale heatmap to a range 0-255
    gb_heatmap = np.uint8(255 * gb_cam)
    
    # We use RGB values of the colormap
    jet_gb_colors = jet(np.arange(256))[:, :3]
    jet_gb_heatmap = jet_colors[gb_heatmap]
    #print(jet_gb_heatmap.shape)
    
    # We create an image with RGB colorized heatmap
    jet_gb_heatmap = keras.preprocessing.image.array_to_img(jet_gb_heatmap)
    jet_gb_heatmap = jet_gb_heatmap.resize((preprocessed_img.shape[1], preprocessed_img.shape[0]))
    jet_gb_heatmap = keras.preprocessing.image.img_to_array(jet_gb_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_gb_img = jet_gb_heatmap * 0.4 + preprocessed_img
    superimposed_gb_img = keras.preprocessing.image.array_to_img(superimposed_gb_img)
    
    
    # Guided Grad CAM
    guided_gradcam = gb_cam * gradcam #gb_cam * gradcam[..., np.newaxis]
    #guided_gradcam = deprocess(guided_gradcam)
    #print(guided_gradcam.shape)
    
    # Visualise Guided GradCAM
    # We rescale heatmap to a range 0-255
    ggb_heatmap = np.uint8(255 * guided_gradcam)
    
    # We use RGB values of the colormap
    jet_ggb_colors = jet(np.arange(256))[:, :3]
    jet_ggb_heatmap = jet_colors[ggb_heatmap]
    
    #print(jet_ggb_heatmap.shape)
    
    # We create an image with RGB colorized heatmap
    jet_ggb_heatmap = keras.preprocessing.image.array_to_img(jet_ggb_heatmap)
    jet_ggb_heatmap = jet_ggb_heatmap.resize((preprocessed_img.shape[1], preprocessed_img.shape[0]))
    jet_ggb_heatmap = keras.preprocessing.image.img_to_array(jet_ggb_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_ggb_img = jet_ggb_heatmap * 0.4 + preprocessed_img
    superimposed_ggb_img = keras.preprocessing.image.array_to_img(superimposed_ggb_img)
    
    if save:
        #jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        #jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        #cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        # Save the superimposed image
        save_path = "gradcam.jpg"
        superimposed_img.save(save_path)
        save_path = "guidedcam.jpg"
        superimposed_gb_img.save(save_path)
        #cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        #cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(141)
        plt.title('Original Image')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        
        plt.subplot(142)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(superimposed_img)
        
        plt.subplot(143)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(superimposed_gb_img)
        
        plt.subplot(144)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(superimposed_ggb_img)
        #plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        st.pyplot()
    
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(241)
        plt.title('Original Image')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        
        plt.subplot(242)
        plt.title('GradCAM')
        plt.axis('off')
        #plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(243)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(gb_cam, cmap='jet', alpha=0.5)
        
        plt.subplot(244)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(guided_gradcam, cmap='jet', alpha=0.5)
        st.pyplot()
        #plt.show()
        
    return gradcam, gb_cam, guided_gradcam

def showResnetCAMs(img, x, GradCAM, GuidedBP, chosen_class, upsample_size):
    cam3 = GradCAM.compute_heatmap(image=x, classIdx=chosen_class, upsample_size=upsample_size)
    #st.write(type(img))
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    #st.write(type(open_cv_image))
    gradcam = overlay_gradCAM(open_cv_image, cam3)
    gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
    # Guided backprop
    gb = GuidedBP.guided_backprop(x, upsample_size)
    gb_im = deprocess_image(gb)
    gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
    # Guided GradCAM
    guided_gradcam = deprocess_image(gb*cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    
    st.header("Displaying details for class-id:{}".format(chosen_class))
    # Display
    plt.figure(figsize=(15, 10))
    plt.subplot(141)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(142)
    plt.title('GradCAM')
    plt.axis('off')
    #plt.imshow(load_image(img_path, preprocess=False))
    plt.imshow(gradcam)

    plt.subplot(143)
    plt.title('Guided Backprop')
    plt.axis('off')
    plt.imshow(gb_im)

    plt.subplot(144)
    plt.title('Guided GradCAM')
    plt.axis('off')
    plt.imshow(guided_gradcam)
    st.pyplot()
    
from tempfile import NamedTemporaryFile
def showIntegratedGradients(model, img, origimg):
    n_steps = 50
    i = 0
    method = "gausslegendre"
    internal_batch_size = 50
    ig  = IntegratedGradients(model,
                              n_steps=n_steps, 
                              method=method,
                              internal_batch_size=internal_batch_size)
    #st.write(img.shape)
    data = img
    data = (data / 255).astype('float32')
    predictions = model(data).numpy().argmax(axis=1)
    explanation = ig.explain(img, 
                             baselines=None, 
                             target=predictions)
    # Get attributions values from the explanation object
    attrs = explanation.attributions
    
    st.header("Integrated Gradient Interpretation")
    st.subheader("Black image baseline")
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    visualize_image_attr(attr=None, original_image=origimg, method='original_image',
                        title='Original Image', plt_fig_axis=(fig, ax[0]), use_pyplot=False);
    
    visualize_image_attr(attr=attrs[i], original_image=data[i], method='blended_heat_map',
                        sign='all', show_colorbar=True, title='Attributions',
                         plt_fig_axis=(fig, ax[1]), use_pyplot=True);
    
    visualize_image_attr(attr=attrs[0], original_image=img[0], method='blended_heat_map',
                        sign='all', show_colorbar=True, title='Overlaid Attributions',
                         plt_fig_axis=(fig, ax[2]), use_pyplot=True);
    st.pyplot()
    
    st.subheader("Random baselines")
    baselines = np.random.random_sample(data.shape)
    explanation = ig.explain(data, 
                         baselines=baselines, 
                         target=predictions)
    attrs = explanation.attributions
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    visualize_image_attr(attr=None, original_image=origimg, method='original_image',
                        title='Original Image', plt_fig_axis=(fig, ax[0]), use_pyplot=False);
    
    visualize_image_attr(attr=attrs[i], original_image=data[i], method='blended_heat_map',
                        sign='all', show_colorbar=True, title='Attributions',
                         plt_fig_axis=(fig, ax[1]), use_pyplot=True);
    
    visualize_image_attr(attr=attrs[i], original_image=origimg, method='blended_heat_map',
                        sign='all', show_colorbar=True, title='Overlaid Attributions',
                         plt_fig_axis=(fig, ax[2]), use_pyplot=True);
    st.pyplot()
    
def Saliency_VanilaResNet50():
    custom_model = st.sidebar.checkbox('Load Fine Tuned Model')
    if custom_model:
        n_classes = 2
        model = load_VanilaResNet50()
    else:
        n_classes = 1000
        model = load_ResNet50V2()
    gradCAM = GradCAM(model=model, layerName="conv5_block3_out")
    guidedBP = GuidedBackprop(model=model,layerName="conv5_block3_out")
    # get the image
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg"]) #"png", "jpg", "jpeg",
    temp_file = NamedTemporaryFile(delete=False)
    if img_file_buffer is not None:
        #mg = Image.open(img_file_buffer)
        temp_file.write(img_file_buffer.getvalue())
        img = keras_image.load_img(temp_file.name)
        img = img.convert('RGB')
        #st.write(type(img))
        img = img.resize((W,H), Image.NEAREST)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet_preprocess_input(x)
        #pred, prob = predict(model,x)
        classIdx = ['0', '1']
        st.header("Predictions")
        if custom_model:
            pred, prob = predict(model,x)
            if pred == '0':
                n = 0
                p = 1
            else:
                n = 1
                p = 0
            selected_classIdx = st.sidebar.selectbox("Choose the label", classIdx, index=n)
            pred_class_idx = [n , p]
            pred_classname = [CUSTOM_CLASS_DECODE[n], CUSTOM_CLASS_DECODE[p]]
            pred_prob = [prob, 1 - prob]
            st.write(pd.DataFrame(zip(pred_class_idx, pred_classname, pred_prob), columns=['Class Id', 'Class Name', 'Probability']))
        else:
            imgidlst = []
            imgclasslst = []
            imgproblst = []
            top_n = 5
            #pred, prob, predclasses, top_5_pred = predict_5(model,x)
            preds = model.predict(x)
            classes = np.argsort(preds[0])[-top_n:][::-1]
            for imgid, imgclass, imgprob in resnet_decode_predictions(preds, top=5)[0]:
                imgidlst.append(imgid)
                imgclasslst.append(imgclass)
                imgproblst.append(imgprob)
            selected_classIdx = st.sidebar.selectbox("Choose the label", classes)                
            st.write(pd.DataFrame(zip(imgidlst,classes,imgclasslst,imgproblst), columns=[ 'Imagenet ID', 'Class Id','Class Name', 'Probability']))
        
        showResnetCAMs(img, x, gradCAM, guidedBP, int(selected_classIdx), (W,H))
        showIntegratedGradients(model, x, img)
    

def Saliency_VGG16():
    # define the model
    model = vgg16_build_model()
    last_conv_layer_name = "block5_conv3"
    classifier_layer_names = [
        "block5_pool",
        "flatten",
        "fc1",
        "fc2",
        "predictions",
    ]
    # get the image
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg"]) #"png", "jpg", "jpeg",
    if img_file_buffer is not None:
        sample_image = Image.open(img_file_buffer)
        sample_image = sample_image.resize((W,H))
        inf_classes = vgg16_compute_predictions(model, sample_image, top_n = 5)
        inf_classes_list = ['-1']
        for c in inf_classes:
            inf_classes_list.append(c)
        selected_class = st.sidebar.selectbox("Choose the label", inf_classes_list)
        gradcam, gb, ggb = compute_vgg16_saliency(model, sample_image, 
                                        layer_name = last_conv_layer_name, classifier_layers = classifier_layer_names, 
                                        cls=selected_class, visualize=True, save=False)
    
def main():
    st.title("Saliency Mapping for CV models")
    # Render the readme as markdown using st.markdown.
    #readme_text = st.markdown("Instructions to run app..")
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Which Model to run")
    app_mode = st.sidebar.selectbox("Choose the model mode",
        ["Model Interpretability", "VGG16", "VanilaResNet50"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "VGG16":
        result_load = Saliency_VGG16()
    elif app_mode == "VanilaResNet50":
        #st.sidebar.warning("Resnet model functionality still in progress.")
        result_load = Saliency_VanilaResNet50()
    #elif app_mode == "ResNet50PlusFC":
    #    result_load = video_analyser()
    else:
        st.sidebar.success('To continue select the "Model".')
        
if __name__ == "__main__":
    main() 

