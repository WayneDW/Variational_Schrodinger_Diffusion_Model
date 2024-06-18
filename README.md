## Variational Schrodinger Diffusion Models (Incomplete version)

- *main* branch: a stable implementation of time-invariant variational scores. 
- *dynamic_module* branch: implementation of time-dependent variational scores.
- *time_series* branch: Applications of VSDM in time series forecasting.
- *image_data*: Image experiment reproduction (TO DO).

## Installation

Following the [link](https://github.com/ghliu/SB-FBSDE), we can install the environment `vsd` using [Anaconda](https://www.anaconda.com/products/individual) as follows
```bash
conda env create --file requirements.yaml python=3
conda activate vsd
```

## Section 1: Consistency check w.r.t. Gaussian Schrodinger bridge

We set beta-r as 0 to fix the hyperparameters of the VP-SDE. We choose $\zeta=1$ in Eqn.(6) since the duality is attainable.

```python
python main.py --problem-name gaussian --num-stage 20 --forward-net Linear --dir gaussian_vsdm_4 --beta-max 4 --beta-r 0. --interact-coef 1
```

<p float="left">
  <img src="figures/Gaussian_Schrodinger_bridge.png" width="400" title="Gaussian Schrodinger bridge" />
  <img src="figures/Variational_Schrodinger_diffusion.png" width="400" title="Variational Schrodinger diffusion" /> 
</p>

## Section 2: Simulation Data (1X on X-axis and 8X on Y-axis)

### Section 2.1: Generation of Non-isotropic shapes

<p float="left">
  <img src="figures/traj_spiral_dsm_10.png" width="270" title="SGM-10" />
  <img src="figures/traj_spiral_dsm_20.png" width="270" title="SGM-20" />
  <img src="figures/spiral_8y_vsdm_10.png" width="270" title="VSDM-10" />
</p>


#### DSM (beta 10 fails)

```python
python main.py --problem-name spiral --num-itr-dsm 100000 --dir spiral_8y_dsm_10 --y-scalar 8 --beta-max 10 --DSM-baseline
```

```python
python main.py --problem-name checkerboard --num-itr-dsm 100000 --dir check_6x_dsm_10 --x-scalar 6 --beta-max 10 --DSM-baseline
```

#### DSM (beta 20 works, but transport is weak)

```python
python main.py --problem-name spiral --num-itr-dsm 100000 --dir spiral_8y_dsm_20 --y-scalar 8 --beta-max 20 --DSM-baseline
```

```python
python main.py --problem-name checkerboard --num-itr-dsm 100000 --dir check_6x_dsm_20 --x-scalar 6 --beta-max 20 --DSM-baseline
```

#### VSDM (beta 10 works via adaptive learning)
```python
python main.py --problem-name spiral --num-itr-dsm 500 --num-stage 200 --forward-net Linear \
               --dir spiral_8y_vsdm_10 --y-scalar 8 --beta-max 10 
```

```python
python main.py --problem-name checkerboard --num-itr-dsm 500 --num-stage 200 --forward-net Linear   \
             --dir check_6x_vsdm_10 --x-scalar 6 --beta-max 10 
```


### Section 2.2: Generation with fewer number of function evaluations (NFEs)

The current code only support NFE=6 (setting interval 108) and 8 (interval 128).

```python
python main.py --problem-name spiral --num-itr-dsm 100000 --dir spiral_dsm_nfe_6 --y-scalar 8 --DSM-baseline --nfe 6
python main.py --problem-name checkerboard --num-itr-dsm 100000 --dir check_dsm_nfe_6 --x-scalar 6 --DSM-baseline --nfe 6 
```

```python
python main.py --problem-name spiral --num-itr-dsm 500 --num-stage 200 --forward-net Linear --dir spiral_vsdm_nfe_6 --y-scalar 8 --interact-coef 0.85 --nfe 6
python main.py --problem-name checkerboard --num-itr-dsm 500 --num-stage 200 --forward-net Linear  --dir check_vsdm_nfe_6 --x-scalar 6  --interact-coef 0.85 --nfe 6
```


### Acknowledgement

https://github.com/ghliu/SB-FBSDE
