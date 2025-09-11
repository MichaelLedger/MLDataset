# [UHD-IQA Benchmark Database​](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html)

A high-resolution NR-IQA database for high-quality authentic photos 

The dataset comprises 6,073 UHD-1 (4k) images. These were selected from the top photos on Pixabay.com based on popularity indicators, out of a total of 1.2 million photos available at the time of indexing. The dataset ensures high photo aesthetics and technical quality, and excludes synthetic images, such as purely computer generated graphics, drawings or those that are heavily edited.

## ​​Features
Resolution: all images are originally larger than UHD-1 (3840x2160 pixels) and are downscaled to a width of 3840px maintaining their aspect ratio.
Authenticity: synthetic images were manually excluded to ensures all are genuine photos.
Reliability: 10 thoroughly selected professional photographers and graphic artists rated each photo twice several days apart, leading to highly reliable mean opinion scores (MOS) based on 20 ratings per image.

## Meta-data: ratings, popularity, indicators, tags (.csv files) 

### uhd-iqa-metadata.csv

image_name: name of the image, based on a dataset specific ID
quality_mos: the quality mean opinion score (MOS) of the image
set: the membership of an image to the training, validation and test sets
subset: membership to the exclusive categories which are only part of the test and validation sets
orig_width, orig_height: original image dimensions​
views, likes, favorites, downloads, comments: popularity indicators from the time the image was indexed
image_bytes: size of the original image in bytes
user_name: the author account name on Pixabay.com
page_url: the address of the image post on Pixabay.com
image_name_orig: the name following the naming convention used on Pixabay.com (used in the tags file below)

### uhd-iqa-tags.csv
image_name: name of the image, based on a Pixabay naming convention 
tag: name of machine tag, per Amazon Rekognition
machine_confidence: confidence in the tag prediction
parents: greater abstraction level tags related to the current

## NOTE:
**Even if the lowest quality images are highly aesthetic, they show degradation that is technical in nature such as blurs, loss of contrast or under-exposure. Removing the confound of aesthetics is a good reason why the database is an interesting challenge for NR-IQA.**
