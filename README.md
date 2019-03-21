# Multiview Marginal Discriminant Projection for Hyperspectral Images Classification

## 1. Introduction

​	This is the source code of our NCIG 2018 paper ["Multiview Marginal Discriminant Projection for Hyperspectral Images Classification"](http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDTEMP&filename=GCTX201806008&uid=WEEvREdxOWJmbC9oM1NjYkZCbDdrdTViZkRDOHpkY2NwZmVOVGQwMWFndzQ=$R1yZ0H6jyaa0en3RxVUd8df-oHi7XMMDo7mtKT6mSmEvTuk11l2gFA!!&v=MDY0NDZZUzdEaDFUM3FUcldNMUZyQ1VSTE9mWk9SdUZ5RG5VcnJPSWk3ZmRyRzRIOW5NcVk5RmJJUjhlWDFMdXg=) ([full-text](https://github.com/Penn000/GMMDP_for_HIC/blob/master/Paper/%E5%9F%BA%E4%BA%8E%E5%A4%9A%E8%A7%86%E5%9B%BE%E8%BE%B9%E7%95%8C%E5%88%A4%E5%88%AB%E6%8A%95%E5%BD%B1%E7%9A%84%E9%AB%98%E5%85%89%E8%B0%B1%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB_%E5%9B%BE%E5%AD%A6%E5%AD%A6%E6%8A%A5.pdf))and JVCI paper ["Graph regularized multiview marginal discriminant projection"](https://www.sciencedirect.com/science/article/pii/S1047320318302451?via%3Dihub) ([full-text](https://github.com/Penn000/GMMDP_for_HIC/blob/master/Paper/GMMDP-JVCI.pdf)). 

​	We employed multiview subspace learning for feature reduction with the  problems of high feature dimension and redundant information of hyperspectral images, and proposed a graph regularized multiview marginal discriminant projection (GMMDP) algorithm. The multiview feature reduction algorithm took the spectral features of each pixels as a view and spatial features as another view, then searched the optimal discriminant common subspace by optimizing the projection direction of each view. 

## 2. Dependency

GMMDP is written by Python 3.6 and following libs are needed:

- sklearn
- numpy
- scipy
- pywt

## 3. Demo

workspace.py is the entrance of program.

```python
print('MvDA', 'indian', 'wavelet')
for i in range(20):
    experiment('MvDA', 'indian', 'wavelet', 20, 0.45)
```

```bash
python3 workspace.py
```

## 4. Reference

Please cite the papers if you use our code.

- GB/T 7714

  > Pan H, He J, Ling Y, et al. Graph Regularized Multiview Marginal Discriminant Projection[J]. Journal of Visual Communication and Image Representation.

- MLA

  > Pan, Heng, et al. "Graph Regularized Multiview Marginal Discriminant Projection." Journal of Visual Communication and Image Representation.

- APA

  > Pan, H., He, J., Ling, Y., Ju, L., & He, G. . Graph regularized multiview marginal discriminant projection. Journal of Visual Communication and Image Representation.



## 5. Contact

Contact me if you have any questions about the code and its execution.

poonhang96@gmail.com