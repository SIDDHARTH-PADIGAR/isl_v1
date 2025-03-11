import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pickle  # <-- NEW import to load model/data

sns.set_theme(style="whitegrid")  # A nice default style from seaborn


def plot_class_distribution(data_dir='./data/'):
    """
    Creates a bar chart showing how many images exist in each class folder.
    Uses Seaborn for a cleaner look.
    """
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_name)
        if os.path.isdir(class_folder):
            num_images = len(os.listdir(class_folder))
            class_counts[class_name] = num_images

    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=counts, palette="rocket", edgecolor='black')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_sample_images(data_dir='./data/', num_samples=5):
    """
    Displays a few sample images (up to num_samples) from different classes.
    Uses a horizontal strip of images for a sophisticated look.
    """
    sample_images = []
    for class_name in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, class_name)
        if os.path.isdir(class_folder):
            folder_images = os.listdir(class_folder)
            if folder_images:  # if not empty
                img_name = folder_images[0]
                img_path = os.path.join(class_folder, img_name)
                sample_images.append((img_path, class_name))

    plt.figure(figsize=(5 * num_samples, 5))
    for i, (img_path, label) in enumerate(sample_images[:num_samples]):
        img = mpimg.imread(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(label, fontsize=12, fontweight='bold')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots a confusion matrix with a sophisticated color palette.
    y_true: list or array of true labels (numeric or string)
    y_pred: list or array of predicted labels
    class_names: list of class names (strings) in the same order as the numeric labels
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = sorted(list(set(y_true)))  # or range(cm.shape[0])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='BuPu', colorbar=True, values_format='d')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Prints a classification report. If class_names is provided,
    the report will be more readable.
    """
    if class_names is not None:
        # Map each label to an index so the report lines up with class_names
        label_to_idx = {label: i for i, label in enumerate(class_names)}
        y_true_idx = [label_to_idx[y] for y in y_true]
        y_pred_idx = [label_to_idx[y] for y in y_pred]
        print(classification_report(y_true_idx, y_pred_idx, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))


def show_training_curves(history):
    """
    If you have a Keras 'history' object from model.fit(...),
    this function plots training vs. validation accuracy and loss.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs, y=acc, label='Train Acc', marker='o')
    sns.lineplot(x=epochs, y=val_acc, label='Val Acc', marker='o')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs, y=loss, label='Train Loss', marker='o')
    sns.lineplot(x=epochs, y=val_loss, label='Val Loss', marker='o')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) Plot class distribution
    plot_class_distribution(data_dir='./data/')

    # 2) Show sample images
    show_sample_images(data_dir='./data/', num_samples=5)

    # === NEW CODE: Evaluate RandomForest on the entire dataset ===
    # 3) Load entire dataset from data.pickle
    with open('data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    X = data_dict['data']
    y_true = data_dict['labels']

    # 4) Load the trained RandomForest model from model.p
    with open('model.p', 'rb') as f:
        saved_model = pickle.load(f)
    model = saved_model['model']

    # 5) Predict on the entire dataset
    y_pred = model.predict(X)

    # 6) Plot confusion matrix & print classification report
    # If your labels are strings like 'A', 'B', 'C', etc., we can pass them directly.
    # If they are numeric, define class_names accordingly or omit the parameter.
    class_names = sorted(list(set(y_true)))  # or define manually if needed

    plot_confusion_matrix(y_true, y_pred, class_names=class_names)
    print_classification_report(y_true, y_pred, class_names=class_names)
