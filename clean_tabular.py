import enum
from collections import Counter, defaultdict
from ntpath import join
import numpy as np
import pandas as pd
import xlsxwriter
import os
import seaborn as sns
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('omw-1.4')
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec
import contractions 
from pathlib import Path
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.io import imread, imshow
from skimage import io
import json
from skimage import img_as_float
from skimage.transform import rescale, resize
from itertools import product
from PIL import Image

class CleanData:
    '''
    Base class for preprocessing image and/or product description tables

    Arguments
    -------------
    level: Level corresponding to minor product category. Must be integer value or 1 or more with a higher value corresponding to a finer classification
    tab_names: Name of json files containing relevant tables
    '''
    def __init__(self, level=1, tab_names = ['Products']) -> None:
        self.tab_names = tab_names # List of json files (without extension) passed in to the class from which dataframes shall be subsequently constructed from 

        # Hard coding the major categories for products uploaded onto site to ensure the encoding remains consistent
        maj_unique_cats = ['Home & Garden ', 'Baby & Kids Stuff ', 'DIY Tools & Materials ', 'Music, Films, Books & Games ', 'Phones, Mobile Phones & Telecoms ', 'Clothes, Footwear & Accessories ', 'Other Goods ', 'Health & Beauty ', 'Sports, Leisure & Travel ', 'Appliances ', 'Computers & Software ','Office Furniture & Equipment ', 'Video Games & Consoles ']
        
        self.major_map_decoder = dict(enumerate(maj_unique_cats)) # Instantiating a dictionary with keys comprissed of the major categories contained in the list defined above and encoding with integer values starting from 0
        self.major_map_encoder = {val: key for key, val in self.major_map_decoder.items()} # Constructing decoder based of encoder defined above by inverting the key value pairs

        # Making a datafiles directory if one does not already exist
        if 'data_files' not in os.listdir():
            os.mkdir(Path(Path.cwd(), 'data_files'))

        self.table_dict = {} # Instantiating emptty dictioanary which shall contain the dataframes constructed using json files

        # Iterates over all json file passed into the list setting the key value of self.table_dict to the table name and the value equal to the datafarme itself constructed using the pd.read_json
        for table in tab_names:
            self.table_dict[table] = pd.read_json(Path(Path.cwd(),'data_files', table+'.json'))
            self.table_dict[table].dropna(inplace = True)
            if 'price' in self.table_dict[table].columns: # Should price be in the json file columns would need preprocessig so that it assumes a form enabling python to recognise it as a float columsn
                self.table_dict[table]['price'] = self.table_dict[table][self.table_dict[table]['price'] != 'N/A'.strip()]['price']  # Removing all columns where price is represented as N/A
                self.table_dict[table]['price'] = self.table_dict[table]['price'].str.replace(',', '').str.strip('£').str.strip(' ').astype(np.float32) # Stripping out the currency symbol, white space and , from price 
                self.table_dict[table] = self.table_dict[table][np.round(self.table_dict[table]['price']) != 0] # Dropping all observations with a price of 0
            if 'category' in self.table_dict[table].columns: # Given the category column is in its raw form represent categories as "major_category"/"minor_category_1"/"minor_category_2"/... applying method attributing a given classification tier its own column
                self.expand_category(df=table, level=level)
                # print(self.table_dict[table].head())
                # print(len(self.table_dict[table]['minor_category_encoded'].value_counts()))
    
    
    def try_merge(self, df_list):
        '''
        Should multiple json files containing either price or product information exist (each containing different observations) this method aggregates
        all observations into one datafraem

        Please note this method is only valid if all tables used to construct are of a single table type i.e., all tables are only price tables or all 
        tables are only images tables

        Arguments
        -------------
        df_list: Must contain dataframes within self.table_dict passed in as a list

        Returns: Dataframe associated with the key 'combined' from table_dict dictionary combining all tables of a given type
        '''
        if isinstance(self.tab_names, str):
            print('Method not valid when class instantiated with tab_names as type string')

        # Iterates over all tables names (keys in self.tick_dict) aggregating the values in  a new dataframe new_df
        else:
            self.new_df = pd.DataFrame(columns = self.table_dict[df_list[0]].columns)
            for i in df_list:
                self.new_df = pd.concat([self.new_df, self.table_dict[i]], axis=0) 
        
        self.table_dict['combined'] = self.new_df # Inserts an additional key into the self.table_dict dictionary with values equal to the newly constructed new_df dataframe
        self.table_dict['combined'].dropna(inplace=True)
        print(self.table_dict['combined'].head())
        return self.table_dict['combined']
    
    def get_na_vals(self, df):
        '''
        Returns a dataframe displaying all observations with na values in any one of its columns 

        Arguments
        -------------
        df: String varaibles equal to one of the keys in self.table_dict

        Returns: Count of all na values occuring within the dataframe
        '''
        print(f'The following NA values exist if dataframe {df}')
        return self.table_dict[df][self.table_dict[df].isna().any(axis=1)]

    def __repr__(self) -> str:
        '''
        Shows the total number of dataframes existing within self.table_dict along with the column name within each of the dataframes
        when using a class instance on the print function
        '''
        if isinstance(self.tab_names, str):
            print(self.df.columns)
            print('\nTable Name: ', self.tab_names, 'With columns:')
            return ' | '.join(self.df.columns)
        else:
            print('\n')
            print('Total of ', f'{len(self.table_dict)} tables')
            return '\n'.join([f'Table Name: {i}: \n' f'Columns | {" | ".join(j.columns)} \n' for i, j in self.table_dict.items()])

    def to_excel(self):
        '''
        Convenience function exporting all dataframes existing within tick_dict onto an excel file with name equal to the its respective key 
        value within self.table_dict onto the data_files directory 

        Retuns: Excel representation of the dataframe within the data_files directory with name equal to the associated key from the table_dict dictionary 
        '''
        for i, j in self.table_dict.items():
            ex_writer = pd.ExcelWriter(f'data_files/{i}.xlsx', engine='xlsxwriter')
            with ex_writer as writer:
                j.to_excel(writer, sheet_name=i)
    
    def cat_set(self, df = 'Products',cat_col = 'major_category'):
        '''
        Dispalys the number of unique categories existing within column
        '''
        return self.table_dict[df][cat_col].nunique()
    
    def expand_category(self, df = 'Products', level=1):
        '''
        Attributes each category tier its own column within the products dataframe up until the tier level specified by the tier argument.
        Given the category column structure of "major_categor"/"minor_category_1"/"minor_category_2"/...,  "major_category" corresponds to  tier level 0,
        "minor_category_1" corresponds to tier level 1 and so on

        Arguments
        -------------
        df: Dataframe/key corresponding to the products dataframe
        level: Tier level as described above, must be integer with maximum value equal to the number of maximum number of tier within the category column

        Returns: Dataframe passed in with additional columns 'major_category_encoded' and 'minor_category_encoded'
        '''
        self.minor_encoder = LabelEncoder() # Instantiates label encoder corresponding to the minro tier cagegory 
        self.table_dict[df]['major_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[0]) # Constructs the major category column
        self.table_dict[df]['minor_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[level]) # Yields the minor category corresponding to the given tier level specified by the user
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['major_category'] != 'N'.strip()] # Removes faultty column existing in given json file
        # print('Encoder', self.major_map_encoder)
        self.table_dict[df]['major_category_encoded'] = self.table_dict[df]['major_category'].map(self.major_map_encoder) # Appllies encoder defined on instantaition of class to major category column to yield the encoded values of such categories
        self.table_dict[df]['minor_category_encoded'] = self.minor_encoder.fit_transform(self.table_dict[df]['minor_category']) # Fits and applies the encoder self.minor_encoder to the minor categories column
        return self.table_dict[df]
    
    def inverse_transform(self, input_array, major_minor = 'minor'):
        '''
        Convenience function for inverse transforming the encoded form of either the major or minor category
        major_minor: Must be one of "major" or "minor" depending on whether the encoded values input correspond to the major or minor category
        input_array: Arry of values to be decoded
        '''
        category_dict = {'major': self.major_encoder, 'minor': self.minor_encoder}
        try:
            return category_dict[major_minor].inverse_transform(input_array)
        except TypeError:
            return category_dict[major_minor].inverse_transform(input_array.numpy())
    
    
    def sum_by_cat(self, df= 'Products', quant = 0.95):
        '''
        Displays summary statistics for teh price decomposed by the product major catgory and illustrates a box-plot showing distribution of price data
        grouped by category and filtered to only those observaations wthin each category with price below the quantile level specified by the user

        Arguments
        -------------
        df: Datafrae (self.table_dict key) containing the relevant price data
        quant: Value between 0 and 1. specifying the quantile level above which all observatoins are ommited from output

        Returns: Cat and whisker plot for each major product category illustrating the price distribution of products within the category 

        '''
        data = self.expand_category(df)
        major = data.groupby('major_category')['price'].describe() # Groups data by major category column and displays summary statistics for price
        print('Price Statistics Grouped by Major Category')
        print(major)
        major_cat_list = major.index.tolist() # Taking index of the summary statistics output to yield the major categories as a list to be iterated over in the next step
        #sns.boxplot(data=data, x = 'major_category', y = 'price')
        products_df = data.loc[:, ['major_category', 'minor_category', 'price']] # Filters original dataframe to only those columns required 

        # Iterates over all the major categories within list "major_cat_list" yielded above, getting the price corresponding to the quantile level specified by the user
        # and plotting box-plot for all observations with a price below the price corresponding to the quantile
        for i in major_cat_list:
            prod_plot = products_df.loc[products_df['major_category'] == i] # Filtering by category being used in current iteration
            print(prod_plot['price'].quantile([quant])) # Getting the price corresponding to the quantile level specified
            print(type(prod_plot['price'].quantile([quant][0]))) 
            print('Number of observations with price more than the 99th quantile: ', len(prod_plot[prod_plot['price'] > prod_plot['price'].quantile([quant][0])])) # Displaying the nu,ber of observations with a price below the quantile level specified
            # sns.boxplot(data=prod_plot, x='major_category', y='price')
            # plt.show()
            sns.boxplot(data=prod_plot[prod_plot['price']<prod_plot['price'].quantile([quant][0])], x = 'major_category', y = 'price') # Constructing box plot
            plt.show()

    def trim_data(self, df= 'Products', quant = 0.95):
        '''
        Trims of all observations with price above the quantile specified by the quant arugment
        '''
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['price'] > self.table_dict[df]['price'].quantile([quant])]
        return self.table_dict[df]

    @classmethod
    def allTables(cls):
        '''
        Alternative means of instantiating class should all json files requied to construct the class exist within the data_files directory.
        Class instantiated using all files with the .json extension within the data_files directory with no arguments requried to be passed in 
        '''
        json_list = []
        json_regex = re.compile(r'(.*).json$')
        for i in os.listdir(Path(Path.cwd(), 'data_files')):
            if re.search(json_regex, i) is not None:
                json_list.append(re.search(json_regex, i).group(1))
        print(json_list)
        return cls(tab_names = json_list)
    
##########################################################################################################################################
##########################################################################################################################################


class CleanImages(CleanData):
    '''
    Class for preprocessing only those JSON files containing image data with basic functionality inheritied fron the CleanData class

    Arguments
    -------------
    level: The category tier level corresponding to which the minor_category column will be constructed
    tab_name: Name of the json files containing image data passed in as a list and without the .json extension
    '''
    def __init__(self, level=1,tab_names=['Images']) -> None:
        super().__init__(level=level, tab_names=tab_names)
        self.df = self.table_dict[tab_names[0]].copy() # self.df desginated the image dataframe
        self.csv_df = None 

    def img_clean_pil(self, size = 512, mode = 'RGB'):
        '''
        Preprocessing of all images within the iamges directory so that they are all of an identical size 

        Arguments
        -------------
        size: Size required for all images to be
        mode: "RGB" if images required to be colored and "L" if images requied to be be grayscale
        '''
        image_re = re.compile(r'(.*)\.jpg') # Compile regex expression for files with jpeg extension
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt', 'images')) # Change working directory to that containing all image file
        # os.chdir(Path(Path.cwd(), 'images'))
        t = 0 # Setting "flag" variables to 0. Subseqeuently used to keep track of how many image have been procesed during iteration process
        for i in os.listdir():
            if re.findall(image_re, i) != []: # If condition on regex expressoin to ensure only jpeg files pre-processed
                try:
                    temp_image = Image.open(i) # get image object using pillow library 
                    black_back = Image.new(size=(size, size), mode=temp_image.mode) # Set a plain black image object of required mode and size onto which image file will subsequently be pasted
                    curr_size = temp_image.size 
                    max_dim = max(temp_image.size) # Yields the legnth equal to the maximum of the image height and width 
                    scale_fact = size / max_dim # Set a scale factor equal to the required size divided by the max_dim variable 
                    resized_image_dim = (int(scale_fact*curr_size[0]), int(scale_fact*curr_size[1])) 
                    updated_image = temp_image.resize(resized_image_dim) # Scales height and width of existing image by the scale factor defined in previous line
                    black_back.paste(updated_image, ((size- resized_image_dim[0])//2, (size- resized_image_dim[1])//2)) # Paste the existing image onto black background image
                    black_back = black_back.convert(mode) # Ensure the mode is of required form 
                    t += 1
                    black_back.save(i) # Saves the image within the same directory overwriting the existing image url 
                except Exception:
                    print(i)
                    with open('invalid_file.json', 'w') as wrong_form:
                        json.dump(i, wrong_form) # Dumps the file from which the error originated into a json file named "invalid_file.json"
                    os.remove(i) # Removes the file yielding the error
                    pass
        print(t) # Should equal to the numbher of images found in directory 
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt')) # Reverts working directory to that set prior to for loop 

    def img_clean_sk(self, normalize = False):
        '''
        Constructs data frame with columns representing each of the following image attributes (1) Image url (2) Numpy represntation of image (3) Image ID 
        (4) The number of channels in image (1 or 3) (5) Image mmode (6) Shape of numpy representation of image 

        Arguments
        -------------
        normalize: If true the numpy representation of the image is bounded to values between 0 and 1

        Returns: image_frame dataframe with additional columns image_id, image_array, image_shape, image_dimension and image_mode
        '''
        image_re = re.compile(r'(.*)\.jpg') # Checks current directory for all images of type jpeg
        img = [] # Instantiate empty lsit for storing image jpeg link
        img_dim_list = [] # Instantiate empty list for storing image dimensions 
        img_id = [] # Instantitae empty list for storing image id
        image_array = [] # Instantiate empty list for storing image in numpy array format
        img_channels = [] # Instantiate empty list for storing number of channels
        img_num_features = [] # Instantiate empty list for storing the number of features corresponding to image (3 for color and 1 for grey)
        img_mode = [] # Instantiate empty list for storing the image mode
        os.chdir(Path(Path.cwd(), 'images'))
        for im in os.listdir():
            if re.findall(image_re, im) != []:
                img.append(im) 
                image = io.imread(im) # Yields the numpy array representation of the image
                if normalize == True:
                    image = img_as_float(image) # If normalization argument set to True sets each pixel to a value betwee 0 and 1 by dividing by 255
                img_id.append(re.search(image_re, im).group(1))
                image_array.append(image)
                img_dim_list.append(image.shape) # 
                if len(image.shape) == 3:
                    img_num_features.append(image.shape[2]) # If image is of type color appends the value 3 
                else:
                    print(im)
                    img_num_features.append(1) # If image of type grayscale appends the value 1
                img_channels.append(len(image.shape))
                img_mode.append(Image.open(im).mode) # Appends the image mode empty list img_mode
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        self.image_frame = pd.DataFrame(data={'image_id': img_id, 'image': img,'image_array': image_array,'image_shape': img_dim_list, 'mode': img_mode}) # Constructs dataframe from lists defined over for iteration corresponding to an indivudal columns
        # print(self.image_frame.head())
        return self.image_frame
    
    def to_excel(self, df):
        '''
        Convenience function exporting dataframe onto an excel file name Clean_Image.xlsx within the data_files directoryt 
        '''
        df.to_excel(Path(Path.cwd(), 'data_files','Cleaned_Images.xlsx'), sheet_name = 'images')

    def merge_images(self):
        '''
        Merges images dataframe with products dataframe for ease of analysis
        '''
        self.df.rename({'id': 'image_id', 'product_id': 'id'}, axis=1, inplace=True) # Renames column id to image_id and product_id to id 
        self.final_df = self.image_frame.merge(self.df, on='image_id', how='inner', validate='one_to_many') # Given both the image and products dataset now have the images uniquely identified under image_id applying inner join on image_id
        # print(self.final_df.head())
        return self.final_df
    
    def edge_detect(self):
        '''
        Uses skimages sobel to apply edge detection to images. Should images be in rgb format the the iamges are first converted to grayscale
        '''
        try:
            self.image_frame['edge_array'] = self.image_frame['image_array'].copy().apply(lambda i: sobel(rgb2gray(i)))
        except: 
            self.image_frame['edge_array'] = self.image_frame['image_array'].copy().apply(lambda i: sobel(i))
        return self.image_frame


    def total_clean(self, normalize=False, mode = 'RGB', size = 224):
        '''
        Convenience function first applying the img_clean_pil method to ensure all images are of identical size whilst maintaining the aspect ratio, 
        then applies img_clean_sk method to yield the images in numpy array form in a separate column along with the image shape and, finally, applies
        the merge_images method to associate each images with the specific product and product categories

        Arguments
        -------------
        normalize: If true the numpy representation of the image is bounded to values between 0 and 1
        mode: If "RGB" images are colored and if "L" images are grayscaled
        '''
        self.img_clean_pil(mode=mode, size=size)
        self.img_clean_sk(normalize=normalize)
        #self.edge_detect()
        self.merge_images()
        return self.final_df
    
    def show_random_images(self, col, size, fig_height= 15, fig_width=10):
        '''
        Randomly displays a set of imags 
        '''
        grid = GridSpec(nrows = size, ncols = size)
        fig = plt.figure(figsize=(fig_height, fig_width))
        for i, j in product(range(size), range(size)):
            fig.add_subplot(grid[i, j]).imshow(self.final_df[col].iloc[np.random.randint(low=0, high=len(self.final_df)-1)])
        plt.show()

    def describe_data(self, df):
        '''
        Outputs summary statistics and column information for image dataframe

        Arguments
        -------------
        df: Number of dataframe contianing images
        '''
        print('\n')
        print('Data frame columnn information')
        print(df.info())
        print('\n')
        print('#'*20)
        print('Dataframe statistical metrics')
        #print(df.describe())
        print('#'*20)
        print('Array and shape')
        print(df['image_shape'].unique())
        print(df['image_shape'].value_counts())


##########################################################################################################################################
##########################################################################################################################################

class MergedData:
    def __init__(self):
        '''
        Class used to clean images and products dataset, yield the respective encoders and merge both datasets on instantiation
        '''
        img_class = CleanImages() # Instantiates CleanImages class
        prod_class = CleanData(tab_names=['Products']) # Cleans the product dataset (automatically done on instantiation of CleanImages class)
        self.major_map_encoder = prod_class.major_map_encoder # Associates encoder for major product categories (automaticallly set on instantiation of CleanImages class) to the self.major_map_encoder
        self.major_map_decoder = prod_class.major_map_decoder # Associates encoder for minor product categories (automaticallly set on instantiation of CleanImages class) to the self.major_map_encoder
        self.prod_frame = prod_class.table_dict['Products'].copy() # Associcates the dataframe associated with the products dataset to the prod_frame attribute
        self.img_df = img_class.total_clean() # Usees CleanImages class to make all images of identical size without altering the aspect ratio and add in additional columns containing the associated numpy arrays for images
        self.merged_frame = self.img_df.merge(self.prod_frame, left_on='id', right_on='id') # Merges prodcuts dataframe and images dataframe 
        # self.merged_frame = self.merged_frame.loc[:, ['image_id', 'product_description']]
    
    def get_val_counts(self):
        return {'products': self.prod_frame, 'images': self.img_df, 'all': self.merged_frame}



if __name__ == '__main__':
    pass
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    # merged = MergedData()
    prod = CleanData(level=1)
    print(prod)
    prod.table_dict['Products'].loc[:, ['minor_category', 'minor_category_encoded']].to_excel(Path(Path.cwd(), 'data_files', 'min_cat.xlsx'))

