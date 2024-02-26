# Malaria Diagnosis Model Analysis

This repository contains code and analysis for a project focused on diagnosing malaria from images of blood cells. The project aims to develop a Convolutional Neural Network (CNN) model to classify blood cells as infected or uninfected with malaria parasites. Various approaches are explored to create an efficient and accurate model for malaria diagnosis.

## Problem Statement

Malaria is a life-threatening disease caused by parasites transmitted through the bite of infected mosquitoes. Early and accurate diagnosis is crucial for effective treatment and prevention of the spread of the disease. Microscopic examination of blood smears remains the gold standard for malaria diagnosis, but it can be time-consuming and requires trained personnel. Automated systems based on machine learning models can assist in rapidly and accurately diagnosing malaria from blood smear images.

The problem addressed in this project is the classification of blood cell images as infected or uninfected with malaria parasites. Given a dataset of blood cell images, the goal is to develop a CNN model that can accurately differentiate between infected and uninfected cells. This model can potentially assist healthcare professionals in quickly identifying malaria cases, leading to timely treatment and better management of the disease.

## Approach

1. **Data Collection**: Gather a dataset of blood cell images containing both infected and uninfected cells. This dataset should be diverse and representative of the variability in real-world samples.

2. **Data Preprocessing**: Preprocess the images to enhance features, remove noise, and normalize pixel values. This step may involve resizing, cropping, and augmenting the images to improve model generalization.

3. **Model Architecture Design**: Experiment with different CNN architectures, including variations of popular models such as VGG, ResNet, and Inception. Fine-tuning pre-trained models on the dataset can also be explored to leverage transfer learning.

4. **Model Training**: Train the designed CNN models on the preprocessed dataset. Utilize techniques such as mini-batch gradient descent, learning rate scheduling, and regularization to optimize model performance and prevent overfitting.

5. **Model Evaluation**: Evaluate the trained models using appropriate metrics such as accuracy, precision, recall, and F1-score. Perform cross-validation and/or holdout validation to assess model generalization and robustness.

6. **Hyperparameter Tuning**: Fine-tune the hyperparameters of the models to further improve performance. This may involve adjusting learning rates, batch sizes, and model architectures based on validation results.

7. **Analysis and Interpretation**: Analyze model predictions, visualize feature maps, and inspect misclassified samples to gain insights into model behavior. This analysis can inform further improvements to the model and highlight areas for future research.

## Approaches
1. ### Simple Sequential Model
   Loss Function: Binary Cross Entropy Loss
   ```Model
   Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_2 (Conv2D)           (None, 222, 222, 6)       168       
                                                                     
     batch_normalization_4 (Bat  (None, 222, 222, 6)       24        
     chNormalization)                                                
                                                                     
     max_pooling2d_2 (MaxPoolin  (None, 111, 111, 6)       0         
     g2D)                                                            
                                                                     
     conv2d_3 (Conv2D)           (None, 109, 109, 16)      880       
                                                                     
     batch_normalization_5 (Bat  (None, 109, 109, 16)      64        
     chNormalization)                                                
                                                                     
     max_pooling2d_3 (MaxPoolin  (None, 54, 54, 16)        0         
     g2D)                                                            
                                                                     
     flatten_1 (Flatten)         (None, 46656)             0         
                                                                     
     dense_3 (Dense)             (None, 100)               4665700   
                                                                     
     batch_normalization_6 (Bat  (None, 100)               400       
     chNormalization)                                                
                                                                     
     dense_4 (Dense)             (None, 10)                1010      
                                                                     
     batch_normalization_7 (Bat  (None, 10)                40        
     chNormalization)                                                
                                                                     
     dense_5 (Dense)             (None, 1)                 11        
                                                                     
    =================================================================
    Total params: 4668297 (17.81 MB)
    Trainable params: 4668033 (17.81 MB)
    Non-trainable params: 264 (1.03 KB)
    _________________________________________________________________
   ```

   *Accuracy*:92.5%


## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/Malaria_Diagnosis_Model.git
cd Malaria_Diagnosis_Model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Explore the notebooks in the `notebooks/` directory to understand the data, model development process, and analysis.

4. Use the provided scripts in the `scripts/` directory to preprocess data, train models, and evaluate model performance.

5. Refer to the reports in the `reports/` directory for detailed analysis and interpretation of results.


