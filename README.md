# Latex Decompiler

A Neural Network to convert images of compiled latex equations into their corrseponding latex code.

## Background

There has been much interest in the conversion of images of mathematical formula into their latex representation. Recently, a commercial tool in the form of [mathpix](https://mathpix.com/), has been implemented to solve this problem. This work is dedicated to recreating that tool with the help of this [paper](https://arxiv.org/pdf/1609.04938v1.pdf).

## The Model
### Attention Plots

A variant of Bahdanau attention was used to help the network attend to the current character. As we can see below, the model has learned to focus on a specific area of the input image and identify the characters from it. It is also evident that it has learned to decode the image from left to right. This ouput was generated after training for 27 epochs and although it gets most characters right, we can still see that it is getting confused on some characters.

The original image

![](src/model_outputs/attention_plots/att-1.png)

The attention at each step

![](src/model_outputs/attention_plots/att-1.gif)

```
Expected:   \pi _ { p } ( s _ { 2 } ) \psi = - \frac { \hbar } 2 \psi \; \; \\mathrm { i } \mathrm { f } \; \; \psi \in { \bf C } ^ { - }

Predicted:  \pi _ { p } ( s _ { 2 } ) \psi = - \frac { \hbar } { 2 } \psi ~ \\mathrm { i } f ~ \mathrm { i f ~ } ~ \psi \in { \bf C } ^ { - }
```

Here is another example where it becomes confused when similar symbols occur multiple times in different places

The original image

![](src/model_outputs/attention_plots/att-2.png)

The attention at each step

![](src/model_outputs/attention_plots/att-2.gif)

```
Expected:   d s ^ { ' 2 } = \frac { 1 } { \operatorname { c o s h } ^ { 2 } \alpha } d s ^ { 2 } \, .

Predicted:  d s ^ { 2 } = \frac { 1 } { \operatorname { c o s h } ^ { 2 } \alpha } d s ^ { 2 } \, .
```
