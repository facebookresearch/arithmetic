PyTorch original implementation of "Making Hard Problems Easier with Custom Data Distributions and Loss Regularization: A Case Study in Modular Arithmetic" (ICML 2025).

**Requirements**
 - Requirements are contained in arithmetic.yml, you can setup a conda env with
```
conda env create -f arithmetic.yml
```
and then run
```
conda activate arithmetic
```

**Dataset creation**
- We provide the dataset generation file in the `/src` folder, in `generation.py` file. 

**Train**
- We provide a `train.py` file to launch training. 
- You can use the flags to tailor the generation / training. For instance

```
python train.py

--N 64 # N dimension
--Q 257 # Q prime
--loss_type custom;0.0001 # Type of loss
--n_enc_layers 4 # Number of layers encoders
--hidden_dim 256 # Embedding dimension
```

- The full list of flag for the model architecture and training parameters can be found in the method `get_parser` of `main.py`

**Reference**  
This code is released under a Creative Commons License, see LICENCE file for more details. If you use this code, consider citing

```    
@article{saxena2024teaching,
  title={Teaching Transformers Modular Arithmetic at Scale},
  author={Saxena, Eshika and Alfarano, Alberto and Wenger, Emily and Lauter, Kristin},
  journal={arXiv preprint arXiv:2410.03569},
  year={2024}
}
```
