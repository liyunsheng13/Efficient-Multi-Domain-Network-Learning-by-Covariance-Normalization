# Efficient Multi-Domain Network Learning by Covariance Normalization (CovNorm) (CVPR 2019)
A [pytorch](http://pytorch.org/) implementation of [Efficient Multi-Domain Network Learning by Covariance Normalization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Efficient_Multi-Domain_Learning_by_Covariance_Normalization_CVPR_2019_paper.pdf).
If you use this code in your research please consider citing
>@inproceedings{li2019efficient,
  title={Efficient Multi-Domain Learning by Covariance Normalization},
  author={Li, Yunsheng and Vasconcelos, Nuno},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5424--5433},
  year={2019}
}
### Requirements

- Hardware: PC with NVIDIA Titan GPU.
- Software: *Ubuntu 16.04*, *CUDA 9.2*, *Anaconda2*, *pytorch 0.4.0*
- Python package
  - `conda install pytorch=0.4.0 torchvision cuda91 -y -c pytorch`
  - `pip install numpy scipy pickle shutil`

### Datasets
* Download the [Caltech256 Dataset](https://drive.google.com/file/d/16RxJyGEtTX7LKB3m_hu8uywtzzNCrH7b/view?usp=sharing) as an example

### Train Residual Adapter
* The initial model can be found:
  * [VGG16-RA](https://drive.google.com/file/d/1hm6VHNwNRpRVGTh8YnWSai4jf2cjttue/view?usp=sharing)
* Training and Evaluation example:

```
python covnorm_train_adpter.py --data-dir /path/to/dataset/256_ObjectCategories \
                               --num-classes 257 \
                               --weight-decay 0.0005 \
                               --gamma 5 \
                               --gpu 0 \
                               --pretrained /path/to/initial-model \
                               --learning-rate 0.001 \
                               --snapshot-dir /path/to/snapshots/Caltech256-RA
```

### Train CovNorm
* Using pretrained Residual Adapter to extract features for residual adapter and intialization to train CovNorm. It can be downloaded:
  * [Caltech256-RA](https://drive.google.com/file/d/1lRpwIZjrdac_0SpUFIwCdoLKMCFj70sQ/view?usp=sharing)
* Extracting features for residual adapter (The pre-extracted features can also be download [Caltech256-feat](https://drive.google.com/file/d/1OCm3uORVIt7rrNSqc4kq8ZK7W8YVqRZ1/view?usp=sharing))

```
python covnorm_feature_extractor.py --data-dir /path/to/dataset/256_ObjectCategories \
                                    --num-classes 257 \
                                    --pretrained-ra /path/to/snapshots/Caltech256-RA \
                                    --gpu 0 \
                                    --length 24384 
```
* Computing whitening and re-coloring matrix

```
python pca/pca.py --root /path/to/Caltech256-feat
```
* Using whitening and re-coloring matrix to help train CovNorm (Thw whitening and re-coloring matrix can also be downloaded [Caltech256-whrc](https://drive.google.com/file/d/1rdoR9XopsLfIq-0lgY0z_tlxle-I3xwv/view?usp=sharing))
  * Training from whiterning and re-coloring matrix:
```
python covnorm_train_adapter_wc.py --data-dir /path/to/dataset/256_ObjectCategories \
                                   --num-classes 257 \
                                   --weight-decay 0.0005 \
                                   --gamma 5 \
                                   --gpu 0 \
                                   --pretrained-ra /path/to/snapshots/Caltech256-RA \
                                   --learning-rate 0.0001 \
                                   --snapshot-dir /path/to/snapshots/Caltech256-CovNorm \
                                   --pca-ratio 0.995
```

### Evaluate CovNorm
* Using pretrained model [Caltech256-CovNorm](https://drive.google.com/file/d/1v9LiWl9gKRV24ZCO5sx5MI5443zqwJC7/view?usp=sharing) to evaluate:
```
python covnorm_eval_adapter_wc.py --data-dir /path/to/dataset/256_ObjectCategories \
                                   --num-classes 257 \
                                   --snapshot-dir /path/to/snapshots/Caltech256-CovNorm \
                                   --pca-ratio 0.995 \
                                   --gpu 0
```

### Other Datasets
The other datasets can be downloaded [Cifar100](https://drive.google.com/file/d/1PmtIQiXiUSKxIPhpEZMVugZg5Hpw0X2l/view?usp=sharing), [SUN397](https://drive.google.com/file/d/1XGdxTWtHXA7LqNRpR9HiF5TXFpQhpJHt/view?usp=sharing), [FGVC](https://drive.google.com/file/d/1bMQyPYYT_RWTlGwmdbZA0_YlAViK4L5y/view?usp=sharing), [Flowers](https://drive.google.com/file/d/1OroXoQqTpatSezxnK_IKyQsDxgTIiB0i/view?usp=sharing), [SVHN](https://drive.google.com/file/d/1Dr1lNetA4n0eStN34CeTAxFc8htMcnkE/view?usp=sharing) and [MITIndoor](https://drive.google.com/file/d/14LkGcCJdKXhoUMRF3iBn051r3lDmEIWv/view?usp=sharing)
