# Dog Breed Classifier

This project is a dog breed classifier that can identify the breed of a dog in an image. If the image contains a human, it will identify the most resembling dog breed. This was a fun project that allowed me to dive deep into the world of Convolutional Neural Networks (CNNs) and transfer learning.

![Sample Output](./images/sample_dog_output.png)

## How it Works

The application works by processing an image through a pipeline of models:

1.  **Human or Dog?** The first step is to determine if the image contains a human or a dog.
    *   **Human Detection:** I used OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces.
    *   **Dog Detection:** To detect dogs, I used a pre-trained VGG-16 model. This model, trained on the massive ImageNet dataset, can identify a wide variety of objects, including many dog breeds.

2.  **Dog Breed Classification:** Once a dog is detected (or a human that looks like a dog!), the image is passed to the breed classifier. I developed two different CNNs for this task:

    *   **A CNN from Scratch:** I first built a CNN from the ground up. This was a great learning experience that helped me understand the fundamentals of CNN architecture. While it achieved a reasonable accuracy, it was clear that a more sophisticated approach was needed for this challenging task.

    *   **A CNN using Transfer Learning:** To achieve a higher accuracy, I used a technique called **transfer learning**.

        **What is Transfer Learning?**

        Imagine you want to become a great chef. Instead of starting from scratch and learning how to boil water, you could learn from a world-renowned chef who has been cooking for decades. You would be "transferring" their knowledge to yourself.

        In the world of AI, transfer learning is very similar. Instead of building a neural network from scratch, we can use a pre-trained model that has already been trained on a huge dataset (like ImageNet, which contains millions of images). This pre-trained model has already learned to recognize a vast number of features, like edges, shapes, and textures.

        For this project, I used the popular **VGG-16 model**. I took the pre-trained VGG-16 model and removed the final layer, which was originally designed to classify 1000 different objects in the ImageNet dataset. I then added my own custom final layer, which is trained to classify the 133 dog breeds in our dataset. This way, I was able to leverage the powerful feature extraction capabilities of the VGG-16 model and achieve a much higher accuracy than my model from scratch.

## Technologies Used

*   **Python**
*   **PyTorch:** The main deep learning framework used for this project.
*   **OpenCV:** Used for human face detection.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For displaying images.

## Setup and Usage

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/dog_breed_classifier_pytorch_project.git
    cd dog_breed_classifier_pytorch_project
    ```
2.  **Download the datasets:**
    *   [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
    *   [Human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
3.  **Open the notebook:**
    ```
    jupyter notebook dog_app.ipynb
    ```
    Follow the instructions in the notebook to see the code in action.

## Results

*   **Model from scratch:** Achieved a test accuracy of **14%**.
*   **Model with transfer learning:** Achieved a test accuracy of **80%**.

This significant improvement in accuracy clearly demonstrates the power of transfer learning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.