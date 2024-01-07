# DL_fracture_recognition
Creation and training of an AI for medical decision support. End of the project : 06/2022.
This project is part of a school project for entrance exams for French universities.

Using Keras (Tensorflow) on python, we create a deep learning model aiming at identifying fracture on medical images (bone X-rays). The model is trained on the MURA dataset from Stanford University (see link at the end). This dataset offers enough data for a consistent training. It tackles various bones, so the solution here focus on each bones one by one. We can easily modify it to train on all bones simultaneously, probably leading to a better generalization but worse accuracies. Pre-processing is applied for better performances, but no data augmentation is done, as the model was really hard to train with my previous computer. The AI is only a binary classification, but we could improve the solution by training the AI to detect ROIs on the images, thus allowing doctors to check the diagnosis.

Inspiration for improvements and source of the dataset : https://stanfordmlgroup.github.io/competitions/mura/
