# 🐯 Lion, Tiger, Cat & Dog Image Classifier using Transfer Learning (ResNet50)

This project applies **Transfer Learning** using the **ResNet50** architecture from Keras Applications to classify images of **Lions**, **Tigers**, **Cats**, and **Dogs**.  
The dataset is created by merging two open-source Kaggle datasets — one for **Lions vs Tigers** and another for **Cats vs Dogs**.

---

## 📂 Dataset Sources

1. **Lions vs Tigers Dataset:**  
   [https://www.kaggle.com/datasets/akrsnv/lions-and-tigers/data](https://www.kaggle.com/datasets/akrsnv/lions-and-tigers/data)

2. **Dogs vs Cats Dataset:**  
   [https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

## 🧠 Project Workflow

1. **Unzip & Organize Datasets**  
   - Extracted both `liontiger.zip` and `dogcat.zip` in Google Colab.  
   - The Cats/Dogs dataset contained a single folder (`PetImages`), while Lions/Tigers had separate `train` and `test` folders.  
   - Merged them into a **4-class dataset**:
     ```
     /merged_train/
         ├── lion/
         ├── tiger/
         ├── cat/
         ├── dog/
     /merged_test/
         ├── lion/
         ├── tiger/
         ├── cat/
         ├── dog/
     ```

2. **Preprocessing**
   - Images were resized to **224×224**.
   - Applied augmentation: rotation, zoom, and horizontal flip.
   - Used `ImageDataGenerator` for train/validation/test splits.

3. **Transfer Learning with ResNet50**
   - Base model: `ResNet50(weights='imagenet', include_top=False)`
   - Unfroze last 30 layers for fine-tuning.
   - Added custom top layers:
     - Global Average Pooling  
     - Dense(256, relu)  
     - Dropout(0.5)  
     - Dense(4, softmax)

4. **Training**
   - Optimizer: **Adam (lr=1e-4)**  
   - Loss: **Categorical Crossentropy**  
   - Metrics: **Accuracy**

---

## 📊 Results

- **Training Epochs:** 6 (stopped early due to time/memory limits — originally planned for 25)
- **Final Test Accuracy:** ≈ **60%**

> 🧩 *Accuracy is lower than expected because training stopped early at 6 epochs instead of 25. With full 25 epochs and more balanced data, accuracy can improve significantly (expected 80%+).*

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| Training Accuracy | ~70% |
| Validation Accuracy | ~60% |
| Test Accuracy | ~60% |

### Accuracy & Loss Curves
- Accuracy and loss graphs were plotted after training.
- A confusion matrix visualized class-level performance.

---

## 🧩 Confusion Matrix
Visualized using `seaborn.heatmap` to show model predictions across four classes (Lion, Tiger, Cat, Dog).

---

## 🛠️ Libraries Used
- `tensorflow`
- `keras`
- `matplotlib`
- `seaborn`
- `sklearn`
- `numpy`
- `os`, `shutil`, `zipfile`

---

## 🚀 Future Improvements
- Train full **25 epochs** for better convergence.
- Add **more balanced samples** for each class.
- Experiment with **VGG19** or **DenseNet121** for comparison.
- Use **Learning Rate Scheduling** and **Early Stopping**.

---

## 👨‍💻 Author
**Ahsan Iftikhar**  
Transfer Learning Project — Deep Learning Lab  
*Colab Implementation: ResNet50 on 4-Class Animal Dataset*
