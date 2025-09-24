环境配置
```shell
conda create -n ropstitch python=3.8.5
conda activate ropstitch
conda install -c conda-forge cudatoolkit=11.6  # 若系统 CUDA 版本一致则不需要再安装
pip install numpy==1.19.5 torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu116/torch_stable.html
pip install setuptools==65.5.1
pip install -r requirements.txt
```

进行与 ChatStitch SV-UDIS 的对比实验：
```shell
cd ./wCoefNet/Codes/
python test_sv_comp.py
```

---

# <p align="center">Robust Image Stitching with Optimal Plane</p>
<p align="center">Lang Nie<sup>∗</sup>, Yuan Mei<sup>†</sup>, Kang Liao<sup>‡</sup>, Yunqiu Xu<sup>§</sup>, Chunyu Lin<sup>¶</sup>, Bin Xiao<sup>∗</sup></p> 
<p align="center"><sup>∗</sup> Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications</p> 
<p align="center"><sup>†</sup> The Hong Kong Polytechnic University</p> 
<p align="center"><sup>‡</sup> Nanyang Technological University</p> 
<p align="center"><sup>§</sup> Zhejiang University</p> 
<p align="center"><sup>¶</sup> Beijing Jiaotong University</p>


![image](./framework.jpg)
## TODO

- **Model**
  - [✖️] Ternary search optimization

## Dataset
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for more details about this dataset. For cross-scenario validation, we have assembled 147 pairs of traditional image stitching pairs, and the associated dataset is available for download [here](https://drive.google.com/file/d/1_F7M7DN7K4BjZPEcez7XS6TUpE3iEX8f/view?usp=drive_link).


## Code
#### Requirement
numpy >= 1.19.5

pytorch >= 1.7.1

scikit-image >= 0.15.0

tensorboard >= 2.9.0

## Training
### Step1: Training the Aligment Model
```
cd ./woCoefNet/Codes/
python train.py
```

### Step2: Training the Iterative Coefficient Generator

```
cd ./wCoefNet/Codes/
python train.py --woCoefNet_path your_model_path
```

## Testing 
Our pretrained models can be available at [Google Drive](https://drive.google.com/drive/folders/1U3kcNM7n_txQ69fjw7wT9EUyAVQZjDxC?usp=drive_link).

```
cd ./wCoefNet/Codes/
python test.py --woCoefNet_path your_model_path
```

## Fine-tuning

```
cd ./wCoefNet/Codes/
python test_finetune.py --woCoefNet_path your_model_path
```

## Meta
If you have any questions about this project, please feel free to drop me an email.

Yuan Mei -- 2551161628@qq.com
