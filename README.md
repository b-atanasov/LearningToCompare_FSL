# Meta-Learning with RelationNet
Reimplementation of PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot Learning part).

The code in this implementation is considerably reorganized, updated to PyTorch 1.2 and simplified. The most notable differences from the original implementation:

* Only one optimizer is used for the whole network. (In the original code there are two separate optimizers for the relation and embedding networks.)
* Custom sampling is removed.
* ResNet18 backbone is added (in similar way to [Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf) but using the ResNet basic block instead of the bottleneck block).
* Category traversal module is added. ([Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf))
* Cross-entropy loss is added.

## Data

Look at the [original repo readme](https://github.com/floodsung/LearningToCompare_FSL#data) for instructions on how to obtain and preprocess the mini-ImageNet. 

## Usage

Train 5-way 5-shot on mini-ImageNet:

```
python main.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --backbone ResNet18 --img_size 224
```

Resume training from the last saved checkpoint:

```
python main.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --resume
```

Test 5-way 5-shot on mini-ImageNet:

```
python main.py --test_folder datas/miniImagenet/test --class_num 5 --sample_num_per_class 5 --test
```

## Reference

[Original Implementation](https://github.com/floodsung/LearningToCompare_FSL)

[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

[Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf)

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)