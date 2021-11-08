# A Brain-inspired Algorithm for Training Highly Sparse Neural Networks
This repository contains code for the paper, "A Brain-inspired Algorithm for Training Highly Sparse Neural Networks" by Zahra Atashgahi, Joost Pieterse, Shiwei Liu, Decebal Constantin Mocanu, Raymond Veldhuis, Mykola Pechenizkiy.
For more information please read the paper at https://arxiv.org/abs/1903.07138. 




### Methodology
![algorithm](https://user-images.githubusercontent.com/18033908/140714651-e548873f-78a1-4579-9ca1-245bb05390ae.JPG)


### Results on highly sparse neural networks
![results](https://user-images.githubusercontent.com/18033908/140714410-fdd79653-9ef3-4954-b78c-651f01657e19.JPG)
Classification accuracy (%) comparison among methods on a highly large and sparse3-layer MLP with a density lower than 0.22%.


### Prerequisites
We run this code on Python 3. Following Python packages have to be installed before executing the project code:
* numpy
* scipy
* sklearn
* Keras


### Reference
If you use this code, please consider citing the following paper:
```
@misc{atashgahi2020quick,
      title={Quick and Robust Feature Selection: the Strength of Energy-efficient Sparse Training for Autoencoders}, 
      author={Zahra Atashgahi and Ghada Sokar and Tim van der Lee and Elena Mocanu and Decebal Constantin Mocanu and Raymond Veldhuis and Mykola Pechenizkiy},
      year={2020},
      eprint={2012.00560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgements
Starting of the code is "Rigging the Lottery: Making All Tickets Winners" which is available at:
https://github.com/google-research/rigl


```
@inproceedings{evci2020rigging,
  title={Rigging the lottery: Making all tickets winners},
  author={Evci, Utku and Gale, Trevor and Menick, Jacob and Castro, Pablo Samuel and Elsen, Erich},
  booktitle={International Conference on Machine Learning},
  pages={2943--2952},
  year={2020},
  organization={PMLR}
}
```

### Contact
email: z.atashgahi@utwente.nl
