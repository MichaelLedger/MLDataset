### AVA Dataset (Aesthetic Visual Analysis)
The AVA dataset can be accessed through the following methods:

## Pretrained Model

[Neural-IMage-Assessment](https://github.com/yunxiaoshi/Neural-IMage-Assessment)
[who can share the pre-train model? ](https://github.com/yunxiaoshi/Neural-IMage-Assessment/issues/40)

The model was trained on the AVA (Aesthetic Visual Analysis) dataset containing 255,500+ images. You can get it from here. Note: there may be some corrupted images in the dataset, remove them first before you start training. Use provided CSVs which have already done this for you.

Dataset is split into 229,981 images for training, 12,691 images for validation and 12,818 images for testing.

An ImageNet pretrained VGG-16 is used as the base network. Should be easy to plug in the other two options (MobileNet and Inception-v2).

The learning rate setting differs from the original paper. Can't seem to get the model to converge using the original params. Also didn't do much hyper-param tuning therefore you could probably get better results. Other settings are all directly mirrored from the paper.

## Download Dataset
1. **Official Download (Recommended)**:
   - Visit [DPChallenge AVA Dataset](http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460)
   - Download using academic torrents (approximately 30GB)
   - Contains:
     - ~255,000 images
     - Aesthetic scores from DPChallenge.com
     - Rich aesthetic annotations
     - Semantic labels

2. **Alternative Sources**:
   - [Kaggle AVA Dataset](https://www.kaggle.com/datasets/kallehjerppe/ava-dataset)
   - Contains preprocessed versions and annotations
   - Easier to download but may not include all original files


```
   ➜  .kaggle pwd
/Users/gavinxiang/.kaggle
➜  .kaggle ls
kaggle.json
➜  .kaggle cat kaggle.json 
{"username":"xianggavin","key":"96be5b4f8bb786eede7cc1b71013fc4d"}% 
```

[ava-aesthetic-visual-assessment](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment)

KAGGLE
```
   cd /Users/gavinxiang/Downloads/MLDataset/AVA && kaggle datasets download -d nicolacarrassi/ava-aesthetic-visual-assessment
```
   
CURL
```
   #!/bin/bash
curl -L -o ~/Downloads/ava-aesthetic-visual-assessment.zip\
  https://www.kaggle.com/api/v1/datasets/download/nicolacarrassi/ava-aesthetic-visual-assessment
  
```

[Hugging Face - AVA](https://huggingface.co/datasets/Iceclear/AVA)

```
$ cd /Users/gavinxiang/Downloads/MLDataset/AVA/huggingface && curl -L -C - "https://huggingface.co/datasets/Iceclear/AVA/resolve/main/AVA_dataset.zip" -o AVA_dataset.zip
```

The curl download is running in the background. It supports resume (-C -) if interrupted.

3. **Dataset Structure**:
   ```
   AVA/
   ├── images/                  # All dataset images
   ├── train.csv               # Training set annotations
   ├── validation.csv          # Validation set annotations
   └── AVA.txt                 # Complete annotations file
   ```

4. **Citation**:
   ```
   @inproceedings{murray2012ava,
     title={AVA: A large-scale database for aesthetic visual analysis},
     author={Murray, Naila and Marchesotti, Luca and Perronnin, Florent},
     booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
     pages={2408--2415},
     year={2012},
     organization={IEEE}
   }
   ```
5. Test
```
head -n 5 /Users/gavinxiang/Downloads/MLDataset/AVA/ground_truth_dataset.csv
```
