# LearningToCompare_FSL
Reimplementation of PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot Learning part).

The code in this implementation is considerably reorganized, updated to PyTorch 1.2 and simplified. The most notable differences from the original implementation:

* Only one optimizer is used for the whole network. (In the original code there are two separate optimizers for the relation and embedding networks.)
* Custom weights initialization is removed.
* Custom sampling is removed.

## Data

Look at the [original repo readme](https://github.com/floodsung/LearningToCompare_FSL#data) for instructions on how to obtain and preprocess the data. 

## Usage

Train 5-way 5-shot on mini-ImageNet:

```
python run.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5
```

Test 5-way 5-shot on mini-ImageNet:

```
python run.py --test_folder datas/miniImagenet/test --class_num 5 --sample_num_per_class 5
```

## Reference

[Original Implementation](https://github.com/floodsung/LearningToCompare_FSL)