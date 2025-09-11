# [KonIQ-10k IQA Database](https://database.mmsp-kn.de/koniq-10k-database.html)

​An ecologically valid image quality assessment database

KonIQ-10k is, at the time of publication, the largest IQA dataset to date consisting of 10,073 quality scored images. This is the first in-the-wild database aiming for ecological validity, with regard to the authenticity of distortions, the diversity of content, and quality-related indicators. Through the use of crowdsourcing, we obtained 1.2 million reliable quality ratings from 1,459 crowd workers, paving the way for more general IQA models.

We introduce a novel, deep learning model (KonCept512), to show an excellent generalization beyond the test set (0.921 SROCC), to the current state-of-the-art database LIVE-in-the-Wild (0.825 SROCC). The model derives its core performance from the InceptionResNet architecture, being trained at a higher resolution than previous models (512x384). A correlation analysis shows that KonCept512 performs similar to having 9 subjective scores for each test image.


## The scores file header

image_name: the image file name
c1-c5: number of ratings for each ACR value 
c_total: total number of judgments per image
MOS: Mean Opinion Scores of the 5-point ACR
​SD: ​Standard Deviation of the MOS
MOS_zscore: ACR scores are first normalised for each user by z-scoring the user ratings

Users exhibiting unusual scoring behaviour are removed. See our paper for more details on how we do reliability screening.

## Example images, with indicators

Mean
Opinion​
Score

Brightness

Colorfulness

Contrast

Sharpness

## Download link
[OSF Storage - KonlQ-10k](https://osf.io/hcsdy/files/osfstorage)

## Unzip
```
unzip 224x224.zip
```
