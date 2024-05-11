# [Re]CUDA: Curriculum of Data Augmentation for Long‐Tailed Recognition

## Abstract
In this reproducibility study, we present our results and experience during replicating the paper,
titled CUDA: Curriculum of Data Augmentation for Long-Tailed Recognition. In real world
scenarios, the dataset we obtain always tend to have an imbalance in the number of samples
available in each class. There will be majority classes having a larger number of samples and
minority classes containing a smaller number of samples. CUDA proposes a class-wise data
augmentation technique which can be used over any existing model to improve the accuracy for
LTR: Long Tailed Recognition. We were able to reproduce a significant part of the results for
dataset Cifar100-LT.

## Organization
The entrire code is presented as a single jupyter notebook master Code which has been briefly documented and cleaned up.
Uncleaned variations of this notebook exist inside each file which shows the exact state of the code when the models were trained.
Code for downloading the depndencies exist within the kaggle code but in chance of any deprecated code dependencies.txt contains the
dependencies and their versions used during training.The readings can be validated by the log files in each folder.The readings for 
validation accuracy has been compiled in the excel sheet "accuracy_readings"

## Training
The model can be trained by altering the Namespace object in the last cell and running the entire notebook with a NVDIA GPU(kaggle).
```python
args=Namespace(network='resnet32', epochs=200, batch_size=128, update_epoch=1,
               lr=0.1, lr_decay=0.01, momentum=0.9, wd=0.0002, nesterov=False,
               scheduler='warmup', warmup=5, aug_prob=0.5, cutout=True, cmo=False,
               posthoc_la=False, cuda=True, aug_type='none', sim_type='none', max_d=30,
               num_test=10, accept_rate=0.6, verbose=False, use_norm=False,
               out='/kaggle/working/log3',
               data_dir='~/dataset/', workers=4, seed='None',
               gpu='0', dataset='cifar100', num_max=500, imb_ratio=100,
               loss_fn='bcl', num_experts=3, ride_distill=False)
```
