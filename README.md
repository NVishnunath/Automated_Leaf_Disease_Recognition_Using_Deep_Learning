# Automated Leaf Disease Recognition Using Deep Learning

## Project Overview:
This project focuses on the automated detection of potato leaf diseases using deep learning techniques. Given the impact of leaf diseases on crop yield and quality, early and accurate detection is crucial. The research compares various convolutional neural network (CNN) architectures—VGG16, VGG19, ResNet50, InceptionV3, and MobileNetV2—to determine the most effective model for disease classification. By leveraging deep learning, this project aims to aid farmers and researchers in efficient disease management and prevention strategies.

## Technology Used:
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow, Keras
- **Models Implemented:** VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2
- **Dataset Source:** Kaggle (Potato Leaf Disease Dataset)
- **Data Processing:** Pandas, NumPy, OpenCV
- **Visualization Tools:** Matplotlib, Seaborn
- **Cloud Integration:** Google Colab, Jupyter Notebook
- **Database & Storage:** PostgreSQL (optional for structured data storage)

## System Design & Workflow:
1. **Data Collection:** Real-time potato leaf images sourced from Kaggle.
2. **Data Processing:**
     - Image resizing and normalization.
     - Data augmentation to improve model robustness.
     - Splitting data into training, validation, and test sets.
3. **Model Training and Evaluation:**
     - Implemented five CNN architectures.
     - Trained models using categorical cross-entropy loss and Adam optimizer.
     - Evaluated performance based on accuracy, precision, and recall.
4. **Comparison of Models:**
     - Accuracy and computational efficiency analyzed.
     - InceptionV3 achieved the highest accuracy (98.05%).

## Results:
- **Best Performing Model:** InceptionV3 with an accuracy of 98.05%.
- **MobileNetV2** also performed well with 95.70% accuracy.
- **VGG16 & VGG19** achieved 94.92% accuracy, demonstrating strong feature extraction capabilities.
- **ResNet50** had the lowest accuracy (44.92%), indicating its inefficiency for this specific task.

### Conclusion:
This research highlights the effectiveness of deep learning in plant disease detection. The InceptionV3 model outperforms other architectures in terms of accuracy, speed, and efficiency. Future work can explore real-time implementation through mobile applications, integration with smart farming technologies, and expansion to other crop diseases. This project provides a strong foundation for AI-driven agricultural advancements, ensuring higher crop yields and sustainable farming practices.
