<div align="center">
  
# $f$-Divergence Based Classification:<br /> Beyond the Use of Cross-Entropy

[Nicola Novello](https://scholar.google.com/citations?user=4PPM0GkAAAAJ&hl=en) and [Andrea M. Tonello](https://scholar.google.com/citations?user=qBiseEsAAAAJ&hl=en)

</div>


Official repository of the paper "$f$-Divergence Based Classification: Beyond the Use of Cross-Entropy " published at ICML 2024. 

> Novel class of objective functions for supervised classification problems, based on the $f$-divergence. We estimate the posterior probability (continuous or discrete) and maximize it to solve classification tasks. 

<div align="center">

[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/nicolaNovello/discriminative-classification-fDiv/blob/main/LICENSE)
[![Hits](https://hits.sh/github.com/nicolaNovello/discriminative-classification-fDiv.svg?label=Visitors&color=30a704)](https://hits.sh/github.com/nicolaNovello/discriminative-classification-fDiv/)

</div>

---

# 📈 Important results from our paper

## Computer Vision

## Signal Detection/Decoding

## Continuous Posterior Esimation

---

# 💻 How to run the code

For the image classification tasks, the file `main.py` runs the experiments. The code runs iterating over multiple random seeds, network architectures and objective functions. They can be set by modifying the lists: 
```
list_cost_func_v = [5] 
random_seeds = [0]
net_architectures = ["ResNet18"] 
dataset_type = "cifar10"
```
where the IDs of the objective functions are:
- 2: GAN
- 3: CE
- 5: SL
- 7: KL with softplus as last activation function
- 9: RKL
- 10: HD
- 12: P
  
while the available network architectures are:
- ResNet18
- PreActResNet18
- MobileNetV2
- VGG
- SimpleDLA
- DenseNet121


For the decoding tasks, the file `main_communications.py` runs the experiments.


---

## 📝 References

If you use your code for your research, please cite our paper:
```

@InProceedings{pmlr-v235-novello24a,
  title = 	 {$f$-Divergence Based Classification: Beyond the Use of Cross-Entropy},
  author =       {Novello, Nicola and Tonello, Andrea M},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {38448--38473},
  year = 	 {2024},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
}

```

## 📋 Acknowledgments

The implementation is based on / inspired by:

- [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)  
- [https://github.com/tonellolab/MIND-neural-decoder](https://github.com/tonellolab/MIND-neural-decoder)

---

## 📧 Contact

[nicola.novello@aau.at](nicola.novello@aau.at)
