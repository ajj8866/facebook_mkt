# Facebook Marketplace Project
### Project Summary and Aim
The aim of this is project is infer as to whether the text within the product description and images, represented by numpy arrays and pytorch tensors can accuratley classify the items into the correct product category 

Scripts Included
- clean_tabular.py
- linear_regression.ipynb
- svm_image_regression.py
- pytorch_image_transfer_classification.py
- pytorch_scratch_classification.py
- combined.py 
- text_model.py 

The data based off of which the model is constructed is stored within the `data_files` directory, specifically the `Products.json` and `Images.json` files, both of which are converted into a dataframe object. The primary variables of concern are the `product_description` column, which is the primary predictor variables when attempting to use NLP to infer which category of products the text most likely refers to and `category` column which constitutes the variable to be predicted. 

Given, in its raw form, each observation within the `category` column comprises of a string which specifies the major product category on the left hand side and proceeds to further the decompose the classification into a finer classfication of minor categories separated by `/` the initial preprocessing entails segregating the category into a major category and a minor category of a level specified by the user. An illustration of the category column before and after is provided below.

<img width="614" alt="image" src="https://user-images.githubusercontent.com/100163231/188771362-68f69f87-537c-42a1-a778-623e313689d2.png">

<img width="350" alt="image" src="https://user-images.githubusercontent.com/100163231/188771429-b582e527-c0ed-4b47-8902-c86a20395cf7.png">


The image model uses the jpeg images in their raw form contained within the image directory to convert them into a consistent size before appending their URL to the images dataframe. 

Though the model as is classifies the major categories an option for predicting the minor category is provided. However, given low frequency with which uniue minor categories appear over the course of the dataset the explanatory power of the model is not only severely limited but a considerable portion of exsiting minor categories are ommitted should they fail to appear a minimum number of times within the dataset. 



linear_regression.ipynb and svm_image_regression are primarily used for initial ananlysis purposes and bear no influence of the main model. 

## clean_tabular.py
Constructs classes used in subsequent scripts pertaining to the preprocessing of data and contains three classes
- CleanData
- CleanImages
- MergedData

### Classes
#### CleanData
Used to extract data containing product details for items listed on facebook marketplace, initially contained in a json file within the data_files directory. The specific json file passed in must be passed in as a list even if only one table is to be used. Given the json files are likely to either contain product information such as the ID, description etc., or image information relevant to such products the class has functionality allowing for merging should any further json files containing new information be provided.  

On instantiation of the class the price column is preprocessed into a format whereby python may interpret it as a float and the categories columns is decomposed into two columns with one correspodning to the major category and the other the minor category with a label encoder fit and applied to both column types. 

Description of the methods are provided as follows:

| Method Name | Method Description |
| --- | --- |
| expand_category | Given the category column in its raw form displays the category type including both the major and minor category this method allocates the major and minor category to individual columns <br /> In addition the encoder (using sklearn's LabelEncoder)  is fit the the transformation applied to both category types|
| traim_data | Trims off all observations with a price above a given quantile specified by the user |
| try_merge | Attempts to merge two dataframes formed as part of the same instance into a single dataframe which is by default named combined, should the individual dataframe be of identical structure |
| get_na_vals | List the observations of a single dataframe instantiated as the class instance |
| __repr__ | Using the print method the class lists all dataframes instantiated as part of the class in addition to the relevant columns within each dataframe |
| to_excel | Transfers all data in the dataframes to an excel table. Mainly for the purposes of convenience in examining data manually should the need arise |
| cat_set | Should the class contain information passed in using the products version of a json file this method lists all unique categories within the dataframe |
| expand_category | Uses the category column to insert two additional columns decomoposing the category into their major and minor components. THough a finer level of detail may be available insertion of any further columns is unlikely to add any additional value as far as analysis is concerned <br /> <br />In addition both the major and minor categories are encoded with numerical representations in order to render them in a format feasible for analysis using machine learning or neural network models |
| inverse_transform | Allows for decoding of the numerical representation of either the major or minor categories |
| trim_data | Distils outliers based on price. By default any items with a price greater than the 95th quantile is removed though this may be adjusted |
| allTables | A class method constructing an instance of the class using all files within the data_files folder |

#### CleanImages
Used specifically for images version of json files with methods used to customise images to different specifications. 

| Method Name | Method Description |
| --- | --- |
| img_clean_pil | Resizes all images to a standard shape and allowing for a change in mode to either 'RGB' or grayscale 'L' |
| img_clean_sk | Constructs dataframe for image version of json file. Prior to inserting data in the dataframe each image is converted to a numpy array with values within the numpy array either being left as is (between 0 and 250) or normalized to between 0 and 1. An additional column contains the shape of each numpy array while another contains the number of dimensions corresponding to the individual numpy arrays |
| to_excel | Similar to `to_excel` in CleanData but overidden so that the excel file named as Cleaned_Images.xlsx |
| merge_images | Merges dataframe constructed using `img_clean_sk` method with dataframes constructed on instantition of json files |
| describe_data | Displays certain statistics for dataframes constructed using the class | 
| total_clean | Successively calls the `img_clean_pil`, `img_clean_sk` and `merge_images` methods to conveniently clean image and merge with dataframe cont |
| edge_detect | Uses `skimge`'s sobel to apply edge detection |
| show_random_images | Illustrates a selection of randomly displayed images |

#### MergedData
Convenience class used to preprocess both the products dataset and image dataset prior to merging both onto a single dataframe. 

Contains the following attributes:
- `prod_frame`: Products dataframe (preprocessed)
- `img_df`: Images datafarme (preprocessed)
- `merged_frame`: Merged images and products dataframe (preprocessed)
- `major_map_encoder`: Encoder for major product category 
- `major_map_decoder`: Decoder for major product category 

<u>Note</u>: Minor category and decoder not set within this class given the specifics would depend on user input when instantiating the mnodel. As such the minor encoder and decoder is set during the model execution phase

## linear_regression.ipynb 
Basic attempt at inferring price using a generic linear regression model with product categories as dummy variables 

Given the simplistic nature of the model the results reflect limited, if any, explanatory power

## svm_image_regression.py 
#### Overview 
Uses sklearn's SVC (support vector classifier) to check whether images, stored as numpy arrays, can help predict the product category it should be classified as. 

#### Process
- Use CleanData and CleanImages class to extract product information in addition to the images corresponding to the products as dataframe
- Merges dataframes extracted in previous steps allowing for a mapping of image numpy arrays to their respective product categories
- Set the explanatory variable as `image_array` and dependent variables as `major_category`
- Datset split into testing set and training set
- Given requirements for running an SVC regression using sklearn the image numpy arrays are stacked vertically as a single numpy array 
- Gridsearch CV for the `C` parameter representing the degree of misclassifications permitted during the training phase
- Model fit and subsequently evaluated on testing dataset using hyperparamter opitimised using GridSearchCV
- Perforamcne on testing data summarised using classficiation report, accuracy score and a heatmap for accuracy score to provide a visual representation of classifications as shown below. Numbers along the diagnol represent the correct predictions made

![image](https://user-images.githubusercontent.com/100163231/169671098-a5434e1b-8f20-44b1-a523-d254292112a9.png)

## pytorch_scratch_classification.py
### Overview
Used to construct the pytorch dataset inheriting from pytorch's `torch.utils.Dataset` class, in addition to a class called Net used for diagnostic purposes to ensure the dataset class works as expected. <br />The class constructed from pytorch's primitive `Dataset` class is used to store the individual sample explanatory variables and their corresponding target variables. The user is subsequently passed into pytorch's `DataLoader` class in the `pytorch_image_transfer_classification` script which iterates over the individual samples stored in the custom class inheriting from pytorch's `torch.utils.Dataset`

#### Dataset Class
Takes in the following arguments: 
- `X`: May be either 'image', in which case the image is passed in raw format or 'image_array' in which case the image is preprocessed via and subsequently converted into a numpy array or 'image_array' in which case the image already preprocessed to a numpy within the clean_tabular script is used
- `y`: Can be either 'major_category_encoded' should the user wish to classify the major categories or 'minor_category_encoded'. Please note the latter is likely to be more inaccurate on account of the sparse nature of certain categories and the added complexity 
- `cutoff_freq`: Only valid if using the 'minor_categories' option for y. Specifies the minimum number of times a minor category must appear in the dataset. Should a given category appear less than the number of times specified the data is dropped from the dataset for analysis
- `img_dir`: Only valid if using the 'image' option for X. Path to directory containing image
- `training_proportion`: The proportion of the datset taken to be the training proportion 
- `is_test`: If True the dataset generated corresponds to the testing data and if set to False the dataset generate corresponds to the training data. Should the user not wish to segregate the data into train and test within the Dataset class set the value to None
- `img_size`: Size of the image required to be used within the model
- `transformer`: Transformations to be applied to the images. <br /> <br /> <b>Please note regardless of the transfomer chose the last two transforamtions passed in must be `[transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]` in order to accommodate for processing of images using numpy arrayss</b>

| Method Name | Method Description |
| --- | --- |
| __init__ | Segregates the data depending on if is_test set to True, False or None. <br /> If y argument is set to minor_category_encoded the dataset is adjusted based on the cutoff_freq argument and a new scaler instantiated <br /> Also sets the minor category encoder and decoder should the category type be minor. Should category type be set to major the encoder may be accessed using the `major_map_encoder` and `major_map_decoder` attributes from the `MergedData` class  |
| __len__ | Sets the length of the datatset |
| __getitem__ | Specifies how the individual observations pertaining to both the target and label are sliced when batching |

#### Net (Demo model)
Demo model used for diganostic purposes pertaining to the Dataset class. The actual model used for deployment purposes is documented within the `pytorch_image_transfer_classification` script

## pytorch_image_transfer_classiciation
### Overview 
- Uses the dataset class from `pytorch_scratch_classification` to construct a dataloader using pytorch's `DataLoader`, transforming and splitting the data into training and testing components during the process
- Sets the optimizers, learning rate scheduler, loss criteria 
- Based on the above paramters adjust resnet50's pretrained model to construct a model attempting to correctly identify the prodct category a given product belongs to given only its image

### Functions and Variables

#### Functions

##### get_label_lim
Stub function used in the event the category type required for classification is the minor category. Based on the minimum number of times a category must appear in the dataset, as stipulated by the `cutoff_lim` argument outputs an integer corresponding to the number of unique minor categories existing within the dataset 

The ouput given by the function is used to calculate the total number of classifications possible for the final layer of the model.

##### get_loader
Uses the Dataset constructed in the `pytorch_scratch_classification` script to construct a dataloader, enabling for the passing of observations using batches into the model during the training and testing phase. Takes in the following arguments:
- `img`: May be either `image` or `image_array` depending on whether the user would like to pass in the images from the data_files directories or as a numpy array using the `MergedData` class (implicitly used in the Dataset class)
- `batch_size`: Number of observations to be passed in during each iteration of the model training and testing loop
- `split_in_dataset`: If set to True splits the dataset into training and testing phase within the Dataset class applying a selection of random flips and rotations to the training portion. If set to False the the `randoom_split` method is applied to the dataloader. In the case of setting the parameter to True the random set of transformations applied to the training phase ensure the issue of overfititing is mitigated and there are, in effect, a greater variety of image to train from but the code will run slower. In the case the parameter is set to False, though the aforementioned issues are not addressed the code will run faster

#### Variables

##### res_model
Establishes resent50's pretrained model as the base model freezing all layers so that the gradients corresponding to such layers are not adjuste over the course of the training process

<img width="483" alt="image" src="https://user-images.githubusercontent.com/100163231/186488027-05895085-aa83-4f47-b352-2f0ad543494b.png">

##### opt
Optimizer to be used in model. By default set to SGD

##### criterion
Loss type to use (by default set to `nn.CrossEntropyLoss()`)

##### scheduler
Learning rate scheduler designed to adjust the laerning rate as the number of epochs progress. By default set to `MultiStepLR` multiplying the initial learning rate by a factor (`gamma` argument) of 0.3 at epochs number 5, 10, 15, 20, 25 and 30

##### class_encoder
Encoder for categories to be predicted within the model. Associated keys (category names) and values (integer respresentation of category names) are stored in `classes` and `class_values` variables, respectively. 

##### classes
List of unique categories to be predicred by the model 

##### class_values
List of encoded values of the categories taken as target variables of the model 

##### plot_classes_preds
Function designed to display the images, actual category classification associated with such images and category classification as predicted by the model (along with associated probabilities) for the first three obsevations passed in for each batch during the training phase of the model <br /> `show_image` and `images_to_proba` are stub functions passed into this function

### Primary Function (train_model)

#### Functionality and Purpose
- Takes in observations passed in using batches output by the `get_loader` function, randomly splitting the dataset into a training and testing phase
- Having split the dataset into a training and testing phase iterates through each dataset
  - For training phase the loss calculated for each batch and weights corresponding to the layer adjusted commensurate with the learning rate 
  - For testing phase the gradients are frozen and associated metrics simply calculated based on the weights derived in the prior training phase
- On the backend tensorboard graphs depicting the progression of loss and accuracy with epochs are generated 
- The 'best' model weights are selected based on the performance of the testing set and eventually returned as the final output 

#### Arguments 
- `model`: Base model to use during the iteration process (by default equal to `res_model`
- `optimizer`: Optimiser to use (by default set to `opt`)
- `loss_type`:  Loss metric to use (by default set to `criterion`)
- `num_epochs`: Number of epochs to use
- `model_scheduler`: Learning rate scheduler to use
- `batch_size`: Batch size to pass in
- `image_type`: Must be one of `"image_array"` or `"image"` depending on model should use actual images or their numpy representation (processed using the `Dataset` class)
- `split_in_datset`: Boolean variable. If set to True the dataset is split within the `Dataset` class and a series of random flips and rotations applied to the training dataset. If set to False the dataset is split within the `get_loader`function and identical transformations applied to both the training and testing set but model runs faster

## text_model
### Overview 
- Sets up a product description class for preprocessing data in the product description column by: 
  -using the `nltk`'s library to lemmitize (converting words like becoming to become, running to run, mice to mouse etc.) individual word, remove punctuation the likes of exclamation marks, colons and fix contraction (words like didn't to did not, wouldn't to would not etc.) and yield all words in lower case so that the model may correcty identify words and the corpus boasts consistent usage of words with identical meaning
   - Obtain a dictionary comprising of all unique words within the product description column and the number of times each unique word appears within the column
   - An additional method (`dataloader_preprocess`) to yield the product description column in a format compatible with the skipgram model. Given the extended period of time this method took the skipgram methodlogy was subsequently discarded in favour of a pre-traned Bert model
 - Set up a dataset called `TextDatasetBert` using a pre-trained Bert model and the tokens corresponding to such a model to obtain 
 
 ### BERT (Bidirectional Encoder Representations from Transformer) Model  
 - Avails of the Recurrnet Neural Network Model (RNN) to construct layers used within the neural network. Unlike CNN's, RNNs focus on estimating sequences such as time series by taking the outputs derived from previous layers as inputs into the subsequent layers. In the current context of deciphering the product description column of the dataset to see if it may be used to predict the relevant product category being referred to the model takes into account the syntatic nature of languages in general, converting each sentence into a vector of fixed length. Intuitively it stands to reason products of a given category would correspond to a product description having a similar sequence of words e.g., games would have words like PS4, computes words like CPU and phone words like iphone. By encoding the sentence containing such words into a vector, the vector representation of such encodings would be similar as given by the cosine of the angle between the two vectors; the cosine of 0 is equal to 1 to two identical vectors would have an angle of 0 between them yielding a similarity of 1. 
 
### Classes
#### ProductDescpClass
Used to preproces the column product description within dataframe to yield individual product descriptions in a format appropriate for them to be passed into any NLP models

| Method Name | Method Description |
| --- | --- |
| clean_prod | Iterates through each individual observation in the product description column applying a series of preprocessing steps such as stripping out uneeded punctuation, fixing contractions (e.g., didn't to did not, couldn't to could not) and lemmitizations (e.g, running to run, fixing to fix) before converting all characters to lower case |
| word_freq | Yields a dictoinary using the individual words occuring over the entirety of the product description column as keys and the number of times each unique word occurs within the dataset as values. <br />If an integer is passed into the `num_items` argument the dictionary is truncated to the integer value specified by the argument resulting in only the most frequently occuring words remaining wthin the dictionary | 
| get_word_set | Returns a tuple comprising of the attributes `full_word_ls` (List of all words appearing in the product description column including included those repeated) `full_word_set` (list of all words appearing in the dataset at least once) and an integer equal to the number of unique words appearing in the product description column + 1 (the plus one added in for convenience when assigning the encoded value for a token appearing outside the predefined vocabulariy size) |
| vocab_encoder | Should the model not have a predefined encoding mechanism of its own this method encodes the product description column using a vocabulary size input by the user as the `vocab_limit` argument or the entirety of vocabulary comprising the product description column should the argument `limit_it` be set to False |
| dataloader_preprocess | Method specifically applied to the skip-gram model of word processing. Preprocesses product description column in a manner whereby the skip-gram model may be directly applied |

#### DescriptionClassifier
Uses sequence of convolutions, dropouts, linear and non-linear layers to train tokenized sentences passed into the model

Takes in the following arguments on instantiation:
- `input_size`: By default 768 but may be adjusted
- `num_classes`: Number of classes model has to predict in final layer (by default equal to 13 which corresponds to the number of major categories

The `forward` method takes in the tokenized representation of sentences to pass through the layyers stipulated in the neural network model 

### Primary Funcation (train_model)
- Uses classes defined in current script to pass in the dataset predictor variables as batches (as specified in the `get_loader` function) to:
  - Adjust weights corresponding to the nodes existing within the model in an attempt to make inferences regarding which particular product category the text within the product description category corresponds to, during the training phase
  - Based on the weights obtained during the training phase, tests the efficacy of applying the weights derived to the next epoch using the testing data
  - The best weights for each layer are selected based on their performance during the training phase and saved under the file name `prodcut_model.pt`
  
 The section after the `__name__=='__main__'` simply declares and input the various arguments into the `train_model` function
 
## Relative Performance
The picture below illustrates the relative performance of the image and text model on tensorboard
- The orange liine represents the performance of only the image model demonstrating an accuracy of over 99% on the training portion of the dataset and approximately 55% for the testing set
- The blue line represents the performance of the text model demonstrating an accuracy of around 95% for the training portion of the dataset and 65% for the testing portion of the dataset

![image](https://user-images.githubusercontent.com/100163231/188768533-68d5612b-4374-44f9-a7eb-339e80868678.png)

It stands to reason rather than using such models in isolation combining the layers from both the image and text models into a single neural network model would add considerable explanatory power, providing the rationale behind the `combined.py` script discussed below. 

## combined.py
This script attempts to draw on the explanatory power impounded by both the prediction model using images to predict product category as well as the prediction model using text to predict product category. <br /> In order to accomplish this both the image model and text model are recycled with one main adjustment; the nerual network layers for both models are truncated so that the final layer (using the prior layer to make the final prediction) is ommitted from the network. The remaining layers are then left as is with the exception of the final linear layers which are resized so that the size of both the output layers stemming from the final output layer are of the same size in order to allocate equal weighting to both models. <br /> In order to accommodate the modeified model the dataset class is ammended so that indexing a single observation yields a tuple comprising of the image index, product description index and the target product description category. <br /> Once the dataloader, dataset and models have been constructed the model (`train_model`) is passed in the training data on which the weightings of the various nodes are adjusted and testing data on which the the efficacy of the model on which the prior epoch was trained is obtained. The weights corresponding to the epoch with the highest accuracy are saved into a file named `image_text_combined.pt`

<b> Please note the current iteration of the combined model does not support passing in `image_array` as an option to the training model's `img` argument. The matter should be resolved in a future iteration </b>

As the tensorboard illustration below demonstrates the combined model results are superior to that of both the training and testing data. However, the testing set results should be taken with a grain of salt given, unliked the isolated image and text models, there is a bit of an overlap between the training and testing data

<img width="240" alt="image" src="https://user-images.githubusercontent.com/100163231/189203319-c90fde3c-0ce2-47ec-a5b8-afe38ec03634.png">

<img width="220" alt="image" src="https://user-images.githubusercontent.com/100163231/189203412-4e5fe5f5-4331-45b6-905c-e534267763c2.png">

## Containerization of Model 
Finally in order for the model to be deployed across machine the model was containerized using the docker specifications in the `Dockerfile` and a docker compose file (`docker-compose.yaml`) constructed for convenience. 

In order to facilitate containerization the python's `fastapi` was used to enable end users using the model on their local machines to test how any image and its corresponding product description would be classified given the weights of the finalized combined model. 

<img width="571" alt="image" src="https://user-images.githubusercontent.com/100163231/189203977-6167155d-bf88-4f26-85e0-b123c83cc0f3.png">

In order to view the API for uploading the relevatn information: 
- Navigate to `htthp://0.0.0.0:8080/docs` on local host
- Click on the `Try it out` button 
<img width="540" alt="image" src="https://user-images.githubusercontent.com/100163231/189219725-5991a1bc-bad9-4a3c-92e5-57991c6a6630.png">
- Upload the relevant image file using the `choose file` and enter the product description and press execute
<img width="567" alt="image" src="https://user-images.githubusercontent.com/100163231/189219949-ff81d4f5-e014-4631-b706-36e3a689563a.png">

The image below illustrats the model correctly predicted the `Computers and Software` category for an IMac listed on the EBay website

<img width="480" alt="image" src="https://user-images.githubusercontent.com/100163231/189220811-4485d51a-5910-467d-839a-4894bba040cf.png">

<img width="620" alt="image" src="https://user-images.githubusercontent.com/100163231/189220870-b45d7fdf-0acf-4a53-88b8-606357bc7cfb.png">

<img width="422" alt="image" src="https://user-images.githubusercontent.com/100163231/189220955-7223203c-3adc-4e04-89a8-4e7f028ab391.png">



A second upload, also from EBay demonstrates the model correctly predicting the cateogry to be `Baby and Kids Stuff`

<img width="304" alt="image" src="https://user-images.githubusercontent.com/100163231/189221154-19183e70-c67d-412a-a8f1-d5480159bc64.png">

<img width="586" alt="image" src="https://user-images.githubusercontent.com/100163231/189221295-dd8aa83f-c335-4075-8e00-1186d2cfbc09.png">

<img width="538" alt="image" src="https://user-images.githubusercontent.com/100163231/189221397-f95e0b59-4d0a-4bba-8a9c-e74b00071910.png">

## AWS EC2 Runtime
Finally the docker file was used to run the model on an EC2 instance. 
![image](https://user-images.githubusercontent.com/100163231/189420165-0457cac0-b5d4-4b07-be5f-35b35e578d52.png)

![image](https://user-images.githubusercontent.com/100163231/189420211-f8bb698e-5e61-44bc-8885-298545910991.png)
![image](https://user-images.githubusercontent.com/100163231/189420329-33f1c42f-09c6-4c7e-91c0-15bbc2669c65.png)
![image](https://user-images.githubusercontent.com/100163231/189420377-f90b0309-0991-448a-b2e7-55b8ce9a778f.png)


