# YOLOv3-object-detector-with-keras
Preparing YOLO v3 Custom training data
Dataset:
  1. If you want to use google openimage dataset (downloaded from google openimage):
     
         a. clone OIDv4_ToolKit-master folder(above) to local. Path must be space free.
         b. create one tensorflow virtual environment in anaconda and activate it.
         c. first fulfill all the requirements by running
         
                   pip install -r requirements.txt
         
         d. Then run
         
                   python main.py downloader --classes Traffic_light Car Bus --type_csv train --limit 300
                   python main.py downloader --classes Traffic_light Car Bus --type_csv test --limit 100
                   
                   (In classes, you have to write categories name whose images you want to download. If any category name is
                    having 2 or more words, then use '_' to join the words.)
                   (While running, you will be asked to download class-descriptions-boxable.csv (contains the name of all 600
                    classes with their corresponding ‘LabelName’), test-annotations-bbox.csv and train-annotations-bbox.csv 
                    (file contains one bounding box (bbox for short) coordinates for one image, and it also has this bbox’s 
                    Label Name and current image’s ID from the validation set of OIDv5) files to OID/csv_folder directory.)
                    When you run this command, OID folder will get created automatically in OIDv4_ToolKit folder with 2 folders
                    in it (csv folder and dataset folder). Dataset folder contains different folder for different categories. 
                    For each category folder, it will have images and label folder. 
                    
         e. Convert Label files to XML files
                If you will open one of labels file you might see class and coordinates of points in type of this:
                "name_of_the_class left top right bottom"
                run this script:
                
                           Python oid_to_pascal_voc_xml.py
                           
                  This script will create same .xml file name as image in a right format that we'll use later.
                If you are testing this script(oid_to_pascal_voc_xml), and starting it from original OIDv4 ToolKit-master path, you 
                should uncomment this line:
                
                              #os.chdir(os.path.join("OID", "Dataset"))
                              
          f. Converting XML to YOLO v3 file structure
                To train a yolo model there is requirements how annotation file should be made:
                Examples
                               path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
                               path/to/img2.jpg 120,300,250,600,2
                               
                To train YOLO v3 object detection model we need annotations file and classes file. Classes and annotations 
                will be created with below script:
                
                             python voc_to_YOLOv3.py
                             
                    In this script, check these 3:
                    
                 1. dataset_train - this is the location of you downloaded images with xml files
                 2. dataset_file - this is the output file, that will be created with prepared annotation for YOLO training;
                 3. classes_file - don't need to change this, this file will be created with all used classes which were in xml file.
                 
              With this script, 2 txt files will be created in OIDv4_ToolKit-master folder
             
  2. If you have your own dataset along with the csv file (containing image name, bounding box coordinates and class of image)
  
          a. convert the csv file to the YOLOv3 format. To do so, run the conversion script:
          
                      python Convert_to_YOLO_format.py
                      
               2 txt files will be created. one is data_train.txt in images folder and another is data_classes.txt in 
               model_weights folder.
  
  3. Manually label images in dataset
  
          a. use Microsoft's Visual Object Tagging Tool (VoTT) to manually label images. To achieve decent results annotate
             at least 100 images. For good results label at least 300 images and for great results label 1000+ images.
             
          b. download and install vott-2.1.0-win32.exe
               1. Create a New Project and call it Annotations. Under Source Connection choose Add Connection and put Images
                  as Display Name. Under Provider choose Local File System and choose path where images are there and then Save
                  Connection. For Target Connection choose the same folder as for Source Connection. Hit Save Project to finish
                  project creation.
               2. Navigate to Export Settings in the sidebar and then change the Provider to Comma Separated Values (CSV), 
                  then hit Save Export Settings.
               3. Labeling:
                  First create a new tag on the right and give it a relevant tag name.
                  Then draw bounding boxes around your objects. You can use the number key 1 to quickly assign the first tag
                  to the current bounding box.
               4. Export result:
                  Once you have labeled enough images press CRTL+E to export the project. You should now see a folder called
                  vott-csv-export in the same path where images were located. Within that folder, you should see a *.csv file
                  called Annotations-export.csv which contains file names and bounding box coordinates.
               5. convert .csv to yolo format:
                  convert the VoTT csv format to the YOLOv3 format. To do so, run the conversion script:
                  
                              python Convert_to_YOLO_format.py

            Till here, data preparation is done. Now, we have to start training the data

TRAINING

        1. clone training folder
        
        2. Download YOLOv3 weights from YOLO website. and Copy downloaded weights file to model_data folder. (The weights are pre-                  trained on the ImageNet 1000 dataset and thus work well for object detection tasks that are very similar to the types                    of images and objects in the ImageNet 1000 dataset.)
        
        3. Convert the Darknet YOLO model to a Keras model:
        
         python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo_weights.h5
         
        4. Now, to start training, There is 2 ways to train custom model:
             1. train_bottleneck.py:
                   Choose this method if you train on CPU, or want to train model faster (lower accuracy model). 
                   Required a lot of RAM and HDD space.
             2. train.py:
                   Choose this method if you train on GPU.
                   
             Changes if using train.py:
                Line 6.  os.environ['CUDA_VISIBLE_DEVICES'] = '0'   // '0' for running with GPU
                                                                    // '-1' for running with CPU
                Line 19. Change annotation_path to your file (generated txt).
                Line 20. Change log_dir, directory where to save trained model and checkpoints.
                Line 21. Change classes_path to your classes file (generated class txt).
                Line 22. anchors_path, don't change this if you don't know what you are doing.
                Line 34. If training new model, leave it as it is "weights_path='model_data/yolo_weights.h5'", otherwise take 
                         previously trained model weights from logs path and put it in modal_data and train again. With this,
                         it will not train from start and start training from the point till which it was trained earlier.
                Line 57. batch_size = 32, try to train with this, if you receive some kind of memory error, decrease this number.
                Line 76. batch_size = 8, same as in Line 57. I tried to train with gtx1080ti, received memory error while using                                  "batch_size = 32", was able to train with 8.
                Lines 63,64 82, 83. Increase epochs count for better model accuracy.
                
             Changes if using train_bottleneck.py:
                Line 6.  os.environ['CUDA_VISIBLE_DEVICES'] = '0'   // '0' for running with GPU
                                                                    // '-1' for running with CPU
                Line 19. Change annotation_path to your file (learned to generate them in previous tutorial).
                Line 20. Change log_dir, directory where to save trained model and checkpoints.
                Line 21. Change classes_path to your classes file (learned to generate them in previous tutorial).
                Line 22. anchors_path, don't change this if you don't know what you are doing.
                Line 30. If training new model, leave it as it is "weights_path='model_data/yolo_weights.h5'", otherwise link your                                checkpoint.
                Lines 72,73 86, 87, 105, 106. Increase epochs count for better model accuracy.
                
             After finishing training in your logs file there should be created new "trained_weights_final.h5" model file. 
             This will be used for custom detection.
             
  TEST Trained Model
  
             1. test it with image_detect.py. Simply change "model_path" (use the trained_weight_final.h5) and "classes_path", to your used files. Then at 
                the end of file you can find "image = 'hydrant.jpg'" line, here just change your test image name.
                
                        python image_detect.py
                        
                   
