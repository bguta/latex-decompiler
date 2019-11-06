# CRNN

A Neural Network to convert images of latex formula into there corrseponding code.

## Background

An Optical character regcogniton is the conversion of images of typed, handwritten or printed text into machine-encoded text

### Showcase

Here is an example of an image:

![alt text](src/model_outputs/test.png)

### Attention Plots

A variant of Bahdanau attention was used to help the network attend to the current character.
![The Image](src/model_outputs/attention_plots/att-1.png)
![The attention output at each step](src/model_outputs/attention_plots/att-1.gif)