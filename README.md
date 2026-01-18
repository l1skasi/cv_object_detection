# cv_object_detection

## data preparation
We utilize the BDD100K dataset. We have downloaded the dataset containing 100k images:
* 100K Images
* Labels

For each image in the "100K Images" set, there is a corresponding JSON file in the "Labels" directory that provide annotations for each object in the image. 

### 100K -> 10K
Training on the full 100k dataset is computationally heavy for us. Therefore, we decided to reduce the dataset to a subset of 10K images.

We compared two sampling methods.

1. First, we randomly selected 10K images.
``` Objects per category:
category
car              101039
traffic sign      34201
traffic light     26648
person            13127 
```

We noticed that data is imbalanced, with cars being the dominant class, which is expected.

2. Next, we selected images by prioritizing those containing rarer categories using stratification.

```
Objects per category:
category
car              102589
traffic sign      34319
traffic light     26695
person            13151
```
Surprisingly, the results were very similar. So we decided to proceed with random image selection to maintain simplicity.

### Final Dataset Structure

We split the 10K image subset into 70/20/10 splits for training, validation, and testing:
* Train: 7,000 images
* Val: 2,000 images
* Test: 1,000 images


Instead of using the original Labels dataset, we now refer to the generated JSON files stored in the  `data/labels` directory. Those files contain only our 4 target classes (car, person, traffic sign, traffic light).
Each CSV file follows the format: `image, category, x1, y1, x2, y2`

**To-DO**: Unzip your downloaded bdd100k_images_100k.zip in `data` folder such that you have `data\bdd100k_images_100k\100k`