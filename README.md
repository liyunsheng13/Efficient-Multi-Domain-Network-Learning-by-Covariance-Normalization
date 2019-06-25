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
* Extracting features for residual adapter (The pre-extracted features can also be download [Caltech256-feat]())

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
