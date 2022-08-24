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

<u>Note</u>: Minor category and decoder not set withiin this class given the specifics would depend on user input when instantiating the mnodel. As such the minor encoder and decoder is set during the model execution phase

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
Used to construct the pytorch dataset inheriting from pytorch's torch.utils.Dataset class, in addition to a class called Net used for diagnostic purposes to ensure the dataset class works as expected. 

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
#### res_model
Establishes resent50's pretrained model as the base model freezing all layers so that the gradients corresponding to such layers are not adjuste over the course of the training process

<img width="483" alt="image" src="https://user-images.githubusercontent.com/100163231/186488027-05895085-aa83-4f47-b352-2f0ad543494b.png">

#### get_label_lim
Stub function used in the event the category type required for classification is the minor category. Based on the minimum number of times a category must appear in the dataset, as stipulated by the `cutoff_lim` argument outputs an integer corresponding to the number of unique minor categories existing within the dataset 

The ouput given by the function is used to calculate the total number of classifications possible for the final layer of the model.

#### get_loader
Uses the Dataset constructed in the `pytorch_scratch_classification` script to construct a dataloader, enabling for the passing of observations using batches into the model during the training and testing phase. Takes in the following arguments:
- `img`: May be either `image` or `image_array` depending on whether the user would like to pass in the images from the data_files directories or as a numpy array using the `MergedData` class (implicitly used in the Dataset class)
- `batch_size`: Number of observations to be passed in during each iteration of the model training and testing loop
- `split_in_dataset`: If set to True splits the dataset into training and testing phase within the Dataset class applying a selection of random flips and rotations to the training portion. If set to False the the `randoom_split` method is applied to the dataloader. In the case of setting the parameter to True the random set of transformations applied to the training phase ensure the issue of overfititing is mitigated and there are, in effect, a greater variety of image to train from but the code will run slower. In the case the parameter is set to False, though the aforementioned issues are not addressed the code will run faster



