

### Suggested citation

Please consider citing our work:

```
@misc{dycl,
      title={Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization}, 
      author={Suofei Zhang and Xinxin Wang and Xiaofu Wu and Quan Zhou and Haifeng Hu},
      year={2025},
      eprint={2506.23077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23077}, 
}

```




## Use dycl

This will create a virtual environment and install the dependencies described in `requirements.txt`:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

WARNING: as of now this code does not work for newer version of `torch`. It only works with `torch==1.8.1`.


## Datasets

We use the following datasets for our paper:


Once extracted the code should work with the base structure of the datasets. You must precise the direction of the dataset to run an experiment in `DyCL_main/config/dataset/da_campus.yaml`:

```
dataset.data_dir=/Path/To/Your/Data/DA_Campus
```


## Run the code

The code uses Hydra for the config. You can override arguments from command line or change a whole config. You can easily add other configs in happier/config.

In the `test_rerank.py` file, at line 156, you can replace the path to the trained model.
### DA_Campus

<details>
  <summary><b>dycl_train</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python run_dycl.py \
'experience.experiment_name=dycl_da_campus' \
'experience.log_dir=experiments/dycl_train' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
experience.accuracy_calculator.with_rerank=False \
experience.max_iter=300 \
experience.warmup_step=5 \
optimizer=da_campus \
model=resnet_ln \
transform=da_campus \
dataset=da_campus \
dataset_test=da_campus \
loss=dycl
```

</details>

<details>
  <summary><b>dycl_rerank</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python test_rerank.py \
'experience.experiment_name=dycl_da_campus' \
'experience.log_dir=experiments/dycl_rerank' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
experience.accuracy_calculator.with_rerank=True \
experience.warmup_step=5 \
optimizer=da_campus \
model=resnet_ln \
transform=da_campus \
dataset=da_campus \
dataset_test=da_campus \
loss=dycl
```

</details>

## Resources

Links to repo with useful features used for this code:

- HAPPIER: https://github.com/elias-ramzi/HAPPIER.git
- PyTorch: https://github.com/pytorch/pytorch
- Pytorch Metric Learning (PML): https://github.com/KevinMusgrave/pytorch-metric-learning
