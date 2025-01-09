# Resilient Missing-Modality MRI Segmentation Based on Mamba State-Space Modeling and Information-Theoretic Criteria
Official implementatation for paper: Resilient Missing-Modality MRI Segmentation Based on Mamba State-Space Modeling and Information-Theoretic Criteria

## Environment
The required libraries are listed in `environment.yml`
```
cond create -n gmd -f environment.yml
```
## Data preparation
download [BraTS18](https://www.med.upenn.edu/sbia/brats2018/registration.html) and modify paths in `mypath.py`

## training 
run `sh cli/train.sh`
## training
run `sh cli/test.sh`
