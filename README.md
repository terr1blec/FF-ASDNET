# FF-ASDNET: Eye-Tracking-Based ASD Detection Model
This repository contains the implementation of the FF-ASDNET model, which is designed for the automated screening of Autism Spectrum Disorder (ASD) using eye-tracking data from emotional perception tasks. The model leverages deep representation learning and multi-scale feature fusion to analyze eye movement trajectories and identify patterns associated with ASD.

## Repository Structure
- **/Dataprocessing**: Contains scripts for processing raw eye-tracking data, including feature extraction and preprocessing.
- **/Local**: Implements the **Gaze Movements (GM)** and **Spatial Attention Distribution (SAD)** feature extraction, focusing on localized eye movement patterns.
- **/Global**: Handles **Temporal Visual Information (TVI)**, capturing global dynamics of eye movements over time.
- **/Modelfusion**: Combines the features from GM, SAD, and TVI using a neural network for classification. Includes the model fusion and training scripts.

## References
This work is based on the paper **"FF-ASDNET: An ASD Detection Model Using Deep Representation Learning and Multi-scale Fusion on Eye Track Data"**.
