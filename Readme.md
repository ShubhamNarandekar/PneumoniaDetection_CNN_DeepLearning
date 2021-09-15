# Pneumonia Detection with chest X-Ray images using CNN

This project is about creating and building a neural network model for classifying the chest x-ray images in one of the two categories. This is an image classification task where the images are classified into one of the two classes that are 'Pneumonia' and 'Normal'. Download the dataset and move it to the project folder.The dataset is available at the following link:
<br> https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Requirements

Install all the requirements specified in the Requirements.txt file. To do so execute the following command:
<br> **pip install -r Requirements.txt**


```python
pip install -r Requirements.txt
```

    Requirement already satisfied: torch in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 1)) (1.7.1+cu110)Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: torchvision in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 2)) (0.8.2+cu110)
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 3)) (0.23.1)
    Requirement already satisfied: tqdm in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 4)) (4.47.0)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 5)) (1.0.5)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 6)) (1.18.5)
    Requirement already satisfied: matplotlib in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 7)) (3.2.2)
    Requirement already satisfied: seaborn in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 8)) (0.11.0)
    Requirement already satisfied: jupyter in c:\programdata\anaconda3\lib\site-packages (from -r Requirements.txt (line 9)) (1.0.0)
    Requirement already satisfied: typing-extensions in c:\programdata\anaconda3\lib\site-packages (from torch->-r Requirements.txt (line 1)) (3.7.4.2)
    Requirement already satisfied: pillow>=4.1.1 in c:\programdata\anaconda3\lib\site-packages (from torchvision->-r Requirements.txt (line 2)) (7.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn->-r Requirements.txt (line 3)) (0.16.0)
    
    Requirement already satisfied: scipy>=0.19.1 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn->-r Requirements.txt (line 3)) (1.5.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn->-r Requirements.txt (line 3)) (2.1.0)
    Requirement already satisfied: pytz>=2017.2 in c:\programdata\anaconda3\lib\site-packages (from pandas->-r Requirements.txt (line 5)) (2020.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in c:\programdata\anaconda3\lib\site-packages (from pandas->-r Requirements.txt (line 5)) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->-r Requirements.txt (line 7)) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->-r Requirements.txt (line 7)) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->-r Requirements.txt (line 7)) (0.10.0)
    Requirement already satisfied: ipywidgets in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (7.5.1)
    Requirement already satisfied: nbconvert in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (5.6.1)
    Requirement already satisfied: notebook in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (6.0.3)
    Requirement already satisfied: jupyter-console in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (6.1.0)
    Requirement already satisfied: ipykernel in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (5.3.2)
    Requirement already satisfied: qtconsole in c:\programdata\anaconda3\lib\site-packages (from jupyter->-r Requirements.txt (line 9)) (4.7.5)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.6.1->pandas->-r Requirements.txt (line 5)) (1.15.0)
    Requirement already satisfied: traitlets>=4.3.1 in c:\programdata\anaconda3\lib\site-packages (from ipywidgets->jupyter->-r Requirements.txt (line 9)) (4.3.3)
    Requirement already satisfied: nbformat>=4.2.0 in c:\programdata\anaconda3\lib\site-packages (from ipywidgets->jupyter->-r Requirements.txt (line 9)) (5.0.7)
    Requirement already satisfied: ipython>=4.0.0; python_version >= "3.3" in c:\programdata\anaconda3\lib\site-packages (from ipywidgets->jupyter->-r Requirements.txt (line 9)) (7.16.1)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in c:\programdata\anaconda3\lib\site-packages (from ipywidgets->jupyter->-r Requirements.txt (line 9)) (3.5.1)
    Requirement already satisfied: jinja2>=2.4 in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (2.11.2)
    Requirement already satisfied: jupyter-core in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (4.6.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (0.8.4)
    Requirement already satisfied: testpath in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (0.4.4)
    Requirement already satisfied: bleach in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (3.1.5)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (0.3)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (1.4.2)
    Requirement already satisfied: pygments in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (2.6.1)
    Requirement already satisfied: defusedxml in c:\programdata\anaconda3\lib\site-packages (from nbconvert->jupyter->-r Requirements.txt (line 9)) (0.6.0)
    Requirement already satisfied: terminado>=0.8.1 in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (0.8.3)
    Requirement already satisfied: prometheus-client in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (0.8.0)
    Requirement already satisfied: ipython-genutils in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (0.2.0)
    Requirement already satisfied: pyzmq>=17 in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (19.0.1)
    Requirement already satisfied: jupyter-client>=5.3.4 in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (6.1.6)
    Requirement already satisfied: tornado>=5.0 in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (6.0.4)
    Requirement already satisfied: Send2Trash in c:\programdata\anaconda3\lib\site-packages (from notebook->jupyter->-r Requirements.txt (line 9)) (1.5.0)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from jupyter-console->jupyter->-r Requirements.txt (line 9)) (3.0.5)
    Requirement already satisfied: qtpy in c:\programdata\anaconda3\lib\site-packages (from qtconsole->jupyter->-r Requirements.txt (line 9)) (1.9.0)
    Requirement already satisfied: decorator in c:\programdata\anaconda3\lib\site-packages (from traitlets>=4.3.1->ipywidgets->jupyter->-r Requirements.txt (line 9)) (4.4.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\programdata\anaconda3\lib\site-packages (from nbformat>=4.2.0->ipywidgets->jupyter->-r Requirements.txt (line 9)) (3.2.0)
    Requirement already satisfied: colorama; sys_platform == "win32" in c:\programdata\anaconda3\lib\site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.4.3)
    Requirement already satisfied: setuptools>=18.5 in c:\programdata\anaconda3\lib\site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (49.2.0.post20200714)
    Requirement already satisfied: jedi>=0.10 in c:\programdata\anaconda3\lib\site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.17.1)
    Requirement already satisfied: pickleshare in c:\programdata\anaconda3\lib\site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.7.5)
    Requirement already satisfied: backcall in c:\programdata\anaconda3\lib\site-packages (from ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\programdata\anaconda3\lib\site-packages (from jinja2>=2.4->nbconvert->jupyter->-r Requirements.txt (line 9)) (1.1.1)
    Requirement already satisfied: pywin32>=1.0; sys_platform == "win32" in c:\programdata\anaconda3\lib\site-packages (from jupyter-core->nbconvert->jupyter->-r Requirements.txt (line 9)) (227)
    Requirement already satisfied: webencodings in c:\programdata\anaconda3\lib\site-packages (from bleach->nbconvert->jupyter->-r Requirements.txt (line 9)) (0.5.1)
    Requirement already satisfied: packaging in c:\programdata\anaconda3\lib\site-packages (from bleach->nbconvert->jupyter->-r Requirements.txt (line 9)) (20.4)
    Requirement already satisfied: wcwidth in c:\programdata\anaconda3\lib\site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->jupyter-console->jupyter->-r Requirements.txt (line 9)) (0.2.5)
    Requirement already satisfied: pyrsistent>=0.14.0 in c:\programdata\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.16.0)
    Requirement already satisfied: attrs>=17.4.0 in c:\programdata\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->jupyter->-r Requirements.txt (line 9)) (19.3.0)
    Requirement already satisfied: parso<0.8.0,>=0.7.0 in c:\programdata\anaconda3\lib\site-packages (from jedi>=0.10->ipython>=4.0.0; python_version >= "3.3"->ipywidgets->jupyter->-r Requirements.txt (line 9)) (0.7.0)
    

### Directory Structure:

#### - **ShubhamNarandekar_20200132**
  - **Checkpoints**: This folder consists of the best checkpoints for each model in my project.
   - proposed_v1_checkpoint.model
   - proposed_v2_checkpoint.model
   - vgg16_v1_checkpoint.model
   - vgg16_v2_checkpoint.model
  - **Results**: This folder consists of all the results from the project.
   - ProposedModel_ConfusionMatrix.png
   - VGG16_ConfusionMatrix.png
   - EvaluationMetrics.png
   - TestingScores.png
   - LossCurve_ProposedModel_Basic.png
   - LossCurve_ProposedModel_Improved.png
   - LossCurve_VGG16_Basic.png
   - LossCurve_VGG16_Improved.png
  - **Models.py**: The models that I have created and the utility methods that are required throughout the project are in this file.
  - **Main.ipynb**: This notebook consists of all the data loading, pre-processing, training the models and saving the best model checkpoints.
  - **Test_Evaluation.ipynb**: This notebook consists of loading all the checkpoints and evaluating the performance of each model on unseen test data.
  - **Requirements.txt**: This file consists of all the required packages and libraries required to run the project.

### Loading the data:
- Make sure that you specify the right file paths for your training, validation and testing datasets from your file directory.
- If you are using Google collab then you first have to upload the datasets on your drive and then when you open the notebook you have to mount your drive so that your notebook can access it. Once your drive in mounted, copy the paths of you train, valid and test folder and paste it accordingly in your code.

### Instructions to run the project
- To train the models again, open the **Main.ipynb** file and run the kernel from beginning and make to sure to specify right paths for your train valid data set.
- To evaluate the models, open **Test_Evaluation.ipynb** file and run the kernel from beginning and make sure to specify right path for your test data set. Also, carefully specify the paths of your saved checkpoints for different models.

### Training and saving the best model

###### Proposed Model (Basic)

![image.png](attachment:image.png)

The model from the epoch 3 is saved as it has the least validation loss.

###### VGG16 (Basic)

![image.png](attachment:image.png)

The model from the epoch 0 is saved as it has the least validation loss.

###### Proposed Model (Improved)

![image.png](attachment:image.png)

The model from the epoch 9 is saved as it has the least validation loss.

###### VGG16 (Improved)

![image.png](attachment:image.png)

The model from the epoch 8 is saved as it has the least validation loss.

### Model Improvements:

###### Proposed Model (Basic and Improved)

Proposed Model (Basic)
![LossCurve_ProposedModel_Basic.png](attachment:LossCurve_ProposedModel_Basic.png)
Proposed Model (Improved)
![LossCurve_ProposedModel_Improved.png](attachment:LossCurve_ProposedModel_Improved.png)

The first image is of the proposed model basic architecture. We can clearly see that validation loss curve is very much high than the training loss which indicates that the model is performaning good on training data but is failing to generalize well on vlidation data. This case is known as overfitting. The second image of the proposed model with imporved architecture in which the issue of overfitting was taken care of by adding more data and applying regularization. We can see in the second image that now our validation loss is almost similar and less than the training.

###### VGG16 (Basic and Improved)

VGG16 (Basic)
![LossCurve_VGG16_Basic.png](attachment:LossCurve_VGG16_Basic.png)
VGG16 (Improved)
![LossCurve_VGG16_Improved.png](attachment:LossCurve_VGG16_Improved.png)

We can clearly see in the first image that the basic VGG16 architecture with only one fully connected layer is performing poor on the validation data thus causing overftting of the model. The second image is of the improved architecture of VGG16 where regualrization techniques like dropouts and batch normalization was added due to which the model is now performing very good on the validation data. The reason for the validation being very low than the training is when we apply dropout, some amount of neurons are set to zero during training which causes the training loss to go up but during validation or testing all the neurons are used due to which the model generalizes very well on validation or testing data.

### Reproducibility

The project can be run automatically without the need of any additional code or changes. The only change needed is the file path for the training, validation and testing folders. To run the project for evaluation purpose, open the **Test_Evaluation.ipynb** file and correctly specifiy the paths for the model checkpoints that are provided. If you want to run the project for training the models again then you just have to open the **Main.ipynb** file and run the kernel from the beginning. After the completion of the training, four different checkpoints will be saved for four different models. You can then use these checkpoints in the **Test_Evaluation.ipynb** file to evaluate your model's performance. The results are reproducible to a very good extent. There might be a very little difference between the results specified in the report and the results achieved on your machine due to different GPU and CPU hardware specifications. I have used Pytorch, it does not gaurantee exact same results on different machines but if you run the project multiple times on the same machine then you will get the same results. The results reported in the project report are from the Google Collab so it is suggested to run the entire project on Google Collab for similar results.
