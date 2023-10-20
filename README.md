# PFST
Official implementation of TGRS paper: "Pseudo Features Guided Self-training for Domain Adaptive Semantic Segmentation of Satellite Images"

## Prepare the Datasets
### Install Dataset4EO
We use Dataset4EO to handle our data loading. Dataset4EO is a composable data loading package based on TorchData. More information can be found at https://github.com/EarthNets/Dataset4EO.

```shell
git clone https://github.com/EarthNets/Dataset4EO.git
cd Dataset4EO
sh install_requirements.sh
pip install -e .
```


### Potsdam and Vaihingen Datasets
1. Download the datasets at 
https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
and https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx

2. Crop the dataset to patches with our provided scripts:
```shell
python tools/convert_datasets/potsdam.py /path/to/your_datasets/Potsdam --tmp_dir ./ --out_dir /path/to/datasets/Potsdam_IRRG_1024_mmlab
python tools/convert_datasets/vaihingen.py /path/to/your_datasets/Vaihingen --tmp_dir ./ --out_dir /path/to/datasets/Vaihingen_IRRG_1024_mmlab
```

3. Set the path to the datasets in the configuration files in ./configures/\_base_/pots_irrg2vaih_irrg.py and ./configures/\_base_/vaih_irrg2pots_irrg.py
```shell
data_root_pots = '/path/to/your_datasets/Potsdam_IRRG_1024_mmlab'
data_root_vaih = '/path/to/your_datasets/Vaih_IRRG_1024_mmlab'
```

### Inria dataset
1. Download the dataset at https://project.inria.fr/aerialimagelabeling/, and put the zip file NEW2-AerialImageDataset.zip in /path/to/your_datasets/Inria

2. Set the path to the datasets in the configuration file in ./configures/\_base_/inria_da.py
```shell
data_root = '/path/to/your_datasets/Inria'
```

3. Dataset4EO will automatically take care of the unzipping and cropping of the dataset.

### SeasonNet dataset
1. Download the dataset at https://zenodo.org/records/5850307. Put at least 4 files 'spring.zip', 'fall.zip', 'meta.csv', and 'splits.zip' in /path/to/your_datasets/SeasonNet

2. Set the path to the datasets in the configuration file in ./configures/\_base_/season_net.py
```shell
data_root = '/path/to/your_datasets/SeasonNet'
```

## Train the PFST model (Optional)
If you want to train the model by youself instead of using our released checkpoints:

1. For the adaptation between two ISPRS datasets:
```shell
python tools/train.py configs/pfst/pfst_pots_irrg2vaih_irrg_deeplabv3plus_r50-d8.py
python tools/train.py configs/pfst/pfst_vaih_irrg2pots_irrg_deeplabv3plus_r50-d8.py
```

2. For Inria datasets:
```shell
python tools/train.py configs/pfst/pfst_inria_da_deeplabv3plus_r50-d8.py
```

3. For SeasonNet datasets:
```shell
python tools/train.py configs/pfst/pfst_season_net_sp2fa_deeplabv3plus_r50-d8.py
```

## Evaluation
### Potsdam IRRG to Vaihingen IRRG
1. A sample checkpoint for Potsdam IRRG to Vaihingen IRRG setting is provided at https://drive.google.com/file/d/1YyLkD-CgZrGVNJpEhArLg2z0iiuEXvL3/view?usp=share_link: 

2. If you are using the provided checkpoint, place it under the folder work_dirs/pfst_pots_irrg2vaih_irrg_deeplabv3plus_r50-d8/, and then:
```shell
python tools/test.py configs/pfst/pfst_pots_irrg2vaih_irrg_deeplabv3plus_r50-d8.py work_dirs/pfst_pots_irrg2vaih_irrg_deeplabv3plus_r50-d8/pfst_pots_irrg2vaih_irrg.pth --work-dir work_dirs/pfst_pots_irrg2vaih_irrg_deeplabv3plus_r50-d8/ --revise_checkpoint_key=True --eval='mIoU'
```

3. If you are using the checkpoint trained locally, simply replace the path to the checkpoint.

Evaluation on the other settings can be conducted with the same manner, yet one will need to train their models locally.

## Acknowledgement
https://github.com/lhoyer/DAFormer

https://github.com/open-mmlab/mmsegmentation
