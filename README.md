# [Re]CUDA: Curriculum of Data Augmentation for Long‐Tailed Recognition

## Abstract
In this reproducibility study, we present our results and experience during replicating the paper, titled CUDA: 
Curriculum of Data Augmentation for Long-Tailed Recognition(Ahn et al., 2023).Traditional datasets used in image 
recognition, such as ImageNet, are often synthetically balanced, meaning each class has an equal number of samples. 
In practical scenarios, datasets frequently exhibit significant class imbalances, with certain classes having a 
disproportionately larger number of samples compared to others. This discrepancy poses a challenge for traditional
image recognition models, as they tend to favor classes with larger sample sizes, leading to poor performance on minority
classes. CUDA proposes a class-wise data augmentation technique which can be used over any existing model to improve the
accuracy for LTR: Long Tailed Recognition. We successfully replicated all of the results pertaining to the long-tailed 
CIFAR-100-LT dataset and extended our analysis to provide deeper insights into how CUDA efficiently tackles class imbalance

## Organization
The entire code is presented as a single jupyter notebook in master Code which has been briefly documented and cleaned up.
Unmodified variations of this notebook exist inside few sections to show the exact state of the code used for training the model.
Code for downloading the depndencies exist within the kaggle code but in chance of any deprecated code please refer dependencies.txt.
The readings can be validated by the log files in each folder.The readings for validation accuracy has been compiled 
in the excel sheet "accuracy_readings"

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
