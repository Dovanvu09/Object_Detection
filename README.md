# Object Detection Algorithms

This repository contains the implementation of various object detection algorithms, including YOLOv1 (from scratch and with augmentation), Faster R-CNN, and DERT. Each of these algorithms is implemented in separate Jupyter notebooks to provide easy access to the methods and the results of the experiments.

## Table of Contents

- [Introduction](#introduction)
- [Notebooks](#notebooks)
  - [DERT](#dert)
  - [Faster R-CNN](#faster-r-cnn)
  - [YOLOv1 Example](#yolov1-example)
  - [YOLOv1 From Scratch](#yolov1-from-scratch)
  - [YOLOv1 With Augmentation](#yolov1-with-augmentation)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Object detection is a crucial task in computer vision where the goal is to detect and localize objects in an image. This repository includes multiple approaches to solving this problem:

1. **YOLOv1**: A real-time object detection model implemented from scratch with optional data augmentation.
2. **Faster R-CNN**: A region-based convolutional neural network for object detection.
3. **DERT**: An object detection transformer model for a more modern approach.

## Notebooks

### DERT

The `DERT` notebook contains the implementation of the DERT model (Detection Transformer). This notebook demonstrates the use of transformers for object detection, which is a relatively new approach to the task.

### Faster R-CNN

The `Faster_R_CNN1` notebook contains the implementation of the Faster R-CNN model, which is a popular approach for region-based object detection. It uses a Region Proposal Network (RPN) to generate regions of interest and then classifies the objects in those regions.

### YOLOv1 Example

The `yolov1_example` notebook provides an example implementation of YOLOv1, explaining the architecture, how to process the data, and train the model. This notebook is intended as an introductory guide for using YOLOv1 in object detection tasks.

### YOLOv1 From Scratch

The `yolov1_scratch` notebook contains a complete implementation of YOLOv1 from scratch, including detailed steps for building the model, training, and evaluating it. This notebook is for those who want to understand the internal workings of YOLOv1 thoroughly.

### YOLOv1 With Augmentation

The `yolov1_scratch_aug` notebook extends the implementation of YOLOv1 with data augmentation techniques. The augmentation aims to improve model generalization by applying transformations such as rotations, color jittering, and other image manipulations.

## Installation

### Prerequisites

- Python 3.7 or above
- PyTorch 1.7.0 or above
- CUDA (optional, for GPU acceleration)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/Dovanvu09/Object_Detection.git
    cd Object_Detection
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Install Jupyter notebook (optional):

    ```sh
    pip install jupyter
    ```

## Usage

### Running Notebooks

- You can open and run each notebook to explore the different object detection methods implemented.
- Use the following command to start Jupyter notebook:

    ```sh
    jupyter notebook
    ```

- Open the notebook of your choice (e.g., `yolov1_example.ipynb`) and run the cells to see how the model is implemented and trained.

### Training

- For training the YOLOv1 model from scratch, open the `yolov1_scratch.ipynb` notebook and run all the cells.
- For training with data augmentation, use the `yolov1_scratch_aug.ipynb` notebook.

### Testing

- You can use the trained model to test object detection on new images by following the instructions in the `Faster_R_CNN1` or `yolov1_example` notebooks.

## Dataset

- This project uses the Pascal VOC dataset for training and evaluation.
- Place the images in the `images/` directory and their corresponding labels in the `labels/` directory.

## Training

1. Open the notebook of the model you want to train (e.g., `yolov1_scratch.ipynb`).
2. Make sure the dataset is prepared and available in the `images` and `labels` folders.
3. Configure the hyperparameters as per your requirements and run all cells.

## Results

- The YOLOv1 model, when trained from scratch, achieves a mean average precision (mAP) of `0.65`.
- Faster R-CNN achieves a mAP of `0.78` on the Pascal VOC dataset.
- Results can be visualized in the notebooks, where sample images with bounding boxes are displayed.

**Sample output:**

![Sample Output](images/sample_output.jpg)

## Contributing

Contributions are welcome! Please create an issue or a pull request for any feature request or bug fix.

1. Fork the repository.
2. Create your branch:

    ```sh
    git checkout -b feature/your-feature
    ```

3. Commit your changes:

    ```sh
    git commit -m "Add some feature"
    ```

4. Push to the branch:

    ```sh
    git push origin feature/your-feature
    ```

5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy Coding!* ✨
