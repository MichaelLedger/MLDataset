### AVA Dataset (Aesthetic Visual Analysis)
The AVA dataset can be accessed through the following methods:

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
