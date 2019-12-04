# Meta-Learning with RelationNet
Reimplementation of PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot Learning part).

The code in this implementation is considerably reorganized, updated to PyTorch 1.2 and simplified. The most notable differences from the original implementation:

* Only one optimizer is used for the whole network. (In the original code there are two separate optimizers for the relation and embedding networks.)
* Custom sampling is removed.
* ResNet18 backbone is added (in similar way to [Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf) but using the ResNet basic block instead of the bottleneck block).
* Category traversal module is added. ([Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf))
* Cross-entropy loss is added.

## Data

Look at the [original repo readme](https://github.com/floodsung/LearningToCompare_FSL#data) for instructions on how to obtain and preprocess the mini-ImageNet. Use [proc_images.py](./datas/miniImagenet/proc_images.py) from this repo instead of the original one because we want to keep the image sizes unchanged instead of resizing them to 84 by 84. 

```bash
cd datas/miniImagenet/
python proc_images.py
```

## Usage

```bash
python main.py -h
usage: main.py [-h] [--train_folder TRAIN_FOLDER] [--test_folder TEST_FOLDER]
               [--backbone {Conv4,ResNet18}] [--enable_ctm]
               [--img_size {84,224}] [--loss_type {mse,cross-entropy}]
               [--class_num CLASS_NUM]
               [--sample_num_per_class SAMPLE_NUM_PER_CLASS]
               [--batch_num_per_class BATCH_NUM_PER_CLASS]
               [--test_batch_num_per_class TEST_BATCH_NUM_PER_CLASS]
               [--train_episodes TRAIN_EPISODES]
               [--test_episodes TEST_EPISODES] [-lr LEARNING_RATE]
               [--disable_cuda] [--input_channels INPUT_CHANNELS] [--resume]
               [--start_episode START_EPISODE] [--test]

Few Shot Image Recognition

optional arguments:
  -h, --help            show this help message and exit
  --train_folder TRAIN_FOLDER
                        path to training data
  --test_folder TEST_FOLDER
                        path to test|validation data
  --backbone {Conv4,ResNet18}
                        feature extractor architecture
  --enable_ctm          add category traversal module
  --img_size {84,224}   input images will be resized to either 84x84 or
                        224x224
  --loss_type {mse,cross-entropy}
                        choose between MSE or cross-entropy loss
  --class_num CLASS_NUM
                        number of classes
  --sample_num_per_class SAMPLE_NUM_PER_CLASS
                        number of images per class in the support set during
                        meta-training
  --batch_num_per_class BATCH_NUM_PER_CLASS
                        number of images per class in the query set during
                        meta-training
  --test_batch_num_per_class TEST_BATCH_NUM_PER_CLASS
                        number of images per class in the query set during
                        meta-testing
  --train_episodes TRAIN_EPISODES
                        number of training episodes
  --test_episodes TEST_EPISODES
                        number of test_episodes
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  --disable_cuda        disable training/inference on gpu
  --input_channels INPUT_CHANNELS
                        input image channels
  --resume              resume training from the last saved checkpoint
  --start_episode START_EPISODE
                        start episode
  --test                load the best saved model and test it on the data in
                        TEST_FOLDER
```





Train 5-way 5-shot on mini-ImageNet:

```bash
python main.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --backbone ResNet18 --img_size 224 --loss_type cross-entropy --enable_ctm 
```

Model checkpoints are saved every 100 episodes during training. If you want to retrain the same model from scratch you will be asked to confirm the deletion of the saved checkpoints:

```bash
python main.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --backbone ResNet18 --img_size 224 --loss_type cross-entropy --enable_ctm 
Are you sure you want to delete the checkpoints in logs/5way_5shot_ResNet18_224_cross-entropy_ctm? (y/n)
```

You can resume from the last checkpoint using `--resume`:

```bash
python main.py --train_folder datas/miniImagenet/train --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --backbone ResNet18 --img_size 224 --loss_type cross-entropy --enable_ctm  --resume
```

During training, every 5000 epochs, the model is validated on data in `--test_folder` and if the model is better than the previous ones, it is saved. To load the best saved model and test it on mini-ImageNet use `--test`:

```bash
python main.py --test_folder datas/miniImagenet/val --class_num 5 --sample_num_per_class 5 --backbone ResNet18 --img_size 224 --loss_type cross-entropy --enable_ctm  --test
```

## Reference

[Original Implementation](https://github.com/floodsung/LearningToCompare_FSL)

[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

[Finding Task-Relevant Features for Few-Shot Learning by Category Traversal](https://arxiv.org/pdf/1905.11116.pdf)

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)