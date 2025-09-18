========================================================================
Perceptual Quality Assessment of Smartphone Photography
PyTorch Version 1.0 by Hanwei Zhu
Copyright(c) 2020 Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang
All Rights Reserved.

----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please refer to the following paper:
#
Y. Fang et al., "Perceptual Quality Assessment of Smartphone Photography" 
in IEEE Conference on Computer Vision and Pattern Recognition, 2020

Kindly report any suggestions or corrections to hanwei.zhu@outlook.com
----------------------------------------------------------------------
Note:
We recommend to directly unzip the compressed files in the Windows OS.
When unzip the database in Linux OS, the following commands may help.
step1: zip -FF SPAQ.z0x --out SPAQ_x.zip

```
zip -FF SPAQ.z01 --out SPAQ_1.zip
Fix archive (-FF) - salvage what can
	zip warning: Missing end (EOCDR) signature - either this archive
                     is not readable or the end is damaged
Is this a single-disk archive?  (y/n): n
```

step2: unzip SPAQ_x.zip

`unzip SPAQ_1.zip`

TestImage Separated Scope in diff zips:
00001.jpg ~ 01684.jpg
01685.jpg ~ 03367.jpg
03368.jpg ~ 05093.jpg
05094.jpg ~ 06823.jpg
06824.jpg ~ 08556.jpg
08557.jpg ~ 10240.jpg
10241.jpg ~ 11125.jpg

Score:
11125 pictures scored in Annotations/MOS\ and\ Image\ attribute\ scores.xlsx

Label:
11125 pictures labeled in Annotations/Scene\ category\ labels.xlsx

EXIF tags:
11125 pictures EXIF in Annotations/EXIF_tags.xlsx

# Low quality dataset from Kaggle
https://www.kaggle.com/datasets/anamikakumari22/spaq-dataset

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("anamikakumari22/spaq-dataset")

print("Path to dataset files:", path)
```
