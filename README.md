# DeepFaceAnnotations

## Introduction
Facial key points prediction is the task where a neural
network predicts the key points in a face image. Facial key
points are relevant for a variety of tasks, such as face filters [11], emotion recognition [5], and so on. As far as I
know most of the pre-trained models can predict only a certain set of key points consist of nose, eyes, lips,eye brows
and lower skin, where the annotations are available [12]. In
this paper we are looking to conduct a way which can predict key points not only for certain face attributes but for all
attributes in the absence of annotations.
## Problem Statement
The task is predicting key points for each attributes of
face by showing the image of the face only. First problem
is we don’t have labeled key points for all attributes of face
and second problem is related to some special attributes for
instance hair, key points only can not be a good representative for the hair and can cause issues like predicting all
hair key points on center of image and so on. So the best
label for this task is attributes masks as input. I am using
ClebAMask-HQ [3] as my input for prediction 448*448 input images and 19 classes of attributes mask.
