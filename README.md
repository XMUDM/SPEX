# SPEX

This repository holds the code for our TOIS paper: ``SPEX: A Generic Framework for Enhancing Neural Social Recommendation`` [[Paper](https://dl.acm.org/doi/abs/10.1145/3473338)]. If you find it is useful for your work, please consider citing our paper:

> Hui Li, Lianyun Li, Guipeng Xv, Chen Lin, Ke Li, and Bingchuan Jiang, "SPEX: A Generic Framework for Enhancing Neural Social Recommendation," ACM Transactions on Information Systems (TOIS), vol. 40, no. 2, pp. 37:1-37:33, 2022.

	@article{LiLXLL22,
	  author    = {Hui Li and
	               Lianyun Li and
	               Guipeng Xv and
	               Chen Lin and
	               Ke Li and
	               Bingchuan Jiang},
	  title     = {SPEX: A Generic Framework for Enhancing Neural Social Recommendation},
	  journal   = {ACM Transactions on Information Systems (TOIS)},
	  year      = {2022},
	  volume    = {40},
	  number    = {2},
	  pages     = {37:1-37:33}
	}


## Baseline:

1. [**NCF**](https://github.com/guoyang9/NCF): Neural Collaborative Filtering. WWW'17.

2. [**NGCF**](https://github.com/liu-jc/PyTorch_NGCF): Neural Graph Collaborative Filtering. SIGIR'19.

3. [**LightGCN**](https://github.com/gusye1234/LightGCN-PyTorch): Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20.

4. [**GraphRec**](https://github.com/wenqifan03/GraphRec-WWW19): Graph Neural Networks for Social Recommendation. WWW'19.

5. [**SAMN**](https://github.com/chenchongthu/SAMN): Social Attentional Memory Network: Modeling Aspect- and Friend-level Differences in Recommendation. WSDM'19.

6. [**DiffNet++**](https://github.com/PeiJieSun/diffnet): DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation. TKDE'20.

7. [**DANSER**](https://github.com/qitianwu/DANSER-WWW-19): Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems. WWW'19.

8. FuseRec: FuseRec: Fusing User and Item Homophily Modeling with Temporal Recommender Systems. DMKD'21.

## Environment:

- Python 3.7 
- Pytorch 1.5.1
- Tensorflow 2.0.0.

## Project structure:
```
│  README.md
│  
├─XXX_SPEX
│  └─code
│  │   │  main_rec.py
│  │   │  main_11.py
│  │   │  main_auto.py
│  │   │  main_auto_expert_s.py
│  │   │  main_cross.py
│  │   │  main_oy.py
│  │   │  
│  │   ├─utility       
│  │   └─utility2
│  │  
│  └─Data
│      ├─epinion2
│      │  ├─rec     
│      │  └─trust       
│      ├─twitter
│      │  ├─rec   
│      │  └─trust      
│      └─weibo
│          ├─rec     
│          └─trust
│              
└─Trust_SPEX
```

## Datasets:

- [**Epinions**](https://www.cse.msu.edu/~tangjili/trust.html)
- [**Weibo**](https://www.aminer.cn/data-sna#Weibo-Net-Tweet)
- [**Twitter**](https://www.aminer.cn/data-sna#Twitter-Dynamic-Net)


## Train:

### 1. Run both recommendation task and path prediction task.

#### Enter the directory (for example, you can do the following to run LightGCN).

    cd SPEX/LightGCN_SPEX/code

#### Train the model (you can use different datasets and gpu id, and choose the number of heads for SPEX).

User Embedding sharing mode: Direct-Sharing 

    python main_11.py  # 1:1 for multi task loss weight
    
    python main_auto.py / main_auto_expert_s.py   # automatically set task weights
        
User Embedding sharing mode: Cross-Stitch 

    python main_cross.py 

User Embedding sharing mode: Shared-Private
    
    python main_oy.py 

### 2. Run the recommendation task only.
    
    python main_rec.py 
    
### 3. Run the path prediction task only.
    
    cd Trust_SPEX/code
    python trust.py 
    
###  Notes:  
To achieve good results, pay attention to the selection of parameters. For instance, LightGCN needs:

    --dropout=1 --keepprob=0.3     

and you will get similar results after about 50 epochs. 

For more details, you can refer to our paper.
