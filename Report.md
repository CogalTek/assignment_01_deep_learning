### **Image Classification using CNNs and Transfer Learning**

---
- Mathieu Rio
- Rémi Maigrot
---
#### **1. Objective**

The goal of this assignment is to evaluate the effectiveness of transfer learning in image classification tasks. We trained a custom CNN to distinguish between cats and dogs (Experiment 1) and then reused this model for different transfer learning scenarios (Experiments 2–4). For all experiments, the same base architecture and training parameters were used (learning rate = 0.0001, 50 epochs).

---

#### **2. Datasets**

- **Cats vs Dogs** dataset (PetImages/) for all evaluations.
    
- **Stanford Dogs** dataset for pretraining the base model in Experiments 2–4.
    

---

#### **3. Experiments Summary**

- **Experiment 1**: Training a CNN from scratch on the cats and dogs dataset.
    
- **Experiment 2**: Transferring a model trained on Stanford Dogs by replacing only the output layer.
    
- **Experiment 3**: Transferring the model but replacing the output layer and first two convolutional layers.
    
- **Experiment 4**: Transferring the model but replacing the output layer and the last two convolutional layers.
    

---

#### **4. Results**

All experiments were run for 50 epochs. Below is a plot of the training accuracy for each experiment:

![[output.png]]

- **Experiment 1** achieved reasonable performance but started from scratch and had a slower learning curve.
    
- **Experiment 2** quickly reached a very high accuracy, showing the effectiveness of transfer learning when the base features are preserved.
    
- **Experiment 3** showed a slight drop in performance compared to Experiment 2, indicating that early layers contain general features useful even across similar domains.
    
- **Experiment 4** performed very similarly to Experiment 2, which suggests that deeper layers might be more domain-specific and can be safely replaced for better task adaptation without a major performance drop.
    

---

#### **5. Project Structure**

The project is organized into clearly separated components for training, evaluation, and demonstration. Below is an overview of the main files and folders:

```Bash
.
├── PetImages
│   ├── Cat
│   ├── Dog
│   └── Models
├── Report.md
├── assets
│   └── output.png
├── experiences
│   ├── demo.py
│   ├── experience_01.py
│   ├── experience_02.py
│   ├── experience_03.py
│   ├── experience_04.py
│   ├── launcher.py
│   ├── load_pet_dataset.py
│   ├── notify.py
│   └── stanford_train.py
├── images de test
│   ├── cat.jpg
│   ├── chaton.jpg
│   ├── coton.jpeg
│   └── golden.jpg
├── stanford_dogs
│   ├── stanford_dogs_model.keras
│   ├── stanford_dogs_model_save.keras
│   ├── stanford_dogs_model_v2.keras
│   ├── stanford_dogs_model_v3.keras
│   ├── stanford_dogs_training_log.csv
└── └── stanford_dogs_training_log_v3.csv
```

---

#### **6. Quick Demo (demo.py)**

The `demo.py` script loads each trained model and runs inference on a few sample images:
```Bash
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 532ms/step

../PetImages/Dog/9.jpg This image is 0.96% cat and 99.04% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step

../PetImages/Dog/4.jpg This image is 0.07% cat and 99.93% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step

../PetImages/Cat/0.jpg This image is 71.04% cat and 28.96% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step

../PetImages/Cat/9.jpg This image is 96.84% cat and 3.16% dog. IsCorrect=True

Model exp 02

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 255ms/step

../PetImages/Dog/9.jpg This image is 0.12% cat and 99.88% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step

../PetImages/Dog/4.jpg This image is 0.00% cat and 100.00% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step

../PetImages/Cat/0.jpg This image is 99.77% cat and 0.23% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step

../PetImages/Cat/9.jpg This image is 100.00% cat and 0.00% dog. IsCorrect=True

Model exp 03

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step

../PetImages/Dog/9.jpg This image is 1.53% cat and 98.47% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step

../PetImages/Dog/4.jpg This image is 0.02% cat and 99.98% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step

../PetImages/Cat/0.jpg This image is 95.38% cat and 4.62% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step

../PetImages/Cat/9.jpg This image is 100.00% cat and 0.00% dog. IsCorrect=True

Model exp 04

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 288ms/step

../PetImages/Dog/9.jpg This image is 0.07% cat and 99.93% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ **0s 34ms/step

../PetImages/Dog/4.jpg This image is 0.01% cat and 99.99% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step

../PetImages/Cat/0.jpg This image is 99.97% cat and 0.03% dog. IsCorrect=True

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step

../PetImages/Cat/9.jpg This image is 99.93% cat and 0.07% dog. IsCorrect=True
```

---
#### **7. Conclusion**

This assignment confirmed the value of transfer learning. Experiment 2 performed best, showing that reusing most of the pretrained model is highly effective. Replacing the first layers (Experiment 3) had a small negative impact, while replacing the last layers (Experiment 4) kept the performance almost intact. This aligns with the intuition that early convolutional layers learn general features like edges, while deeper ones capture task-specific representations.

---

#### **8. Additional Notes**

- All models were tested on a consistent subset of the PetImages dataset.
    
- Accuracy results were evaluated using actual image predictions (e.g., “99.97% cat – IsCorrect=True”), and performance was consistent across examples.