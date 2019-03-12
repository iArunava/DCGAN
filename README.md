# DCGAN

In this repository, I have reproduced the DCGAN paper. The paper can be found here: [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
The implementation is done using PyTorch. I have also released pretrained models.
Also, I am publishing a colab notebook, with which you can reproduce the entire model in just one go. And with the help
of the pretrained models, you can also, see amazing results, right in your browser and start playing with it.

## How to use

0. Go to the Colab Notebook: [DCGAN Colab](https://colab.research.google.com/github/iArunava/DCGAN/blob/master/DCGAN.ipynb)

---

0. Clone the repository and cd into it
```
git clone https://github.com/iArunava/DCGAN.git
cd DCGAN/
```

1. Start training the model
```
# To train the model of the Oxford Flowers Dataset
python3 init.py --mode train -dt flowers


# To train the model of the Stanford Cars Dataset
python3 init.py --mode train -dt cars


# To train the model of the Dogs Dataset
python3 init.py --mode train -dt dogs
```

2. Test the model using pretrained model
```
python3 init.py --mode predict -gpath /path/to/pretrained/generator/model.pth
```
Note: You can download the pretrained model, the links are available in the readme of the flowers, cars directories under 
results directories.

## Samples generated using DCGAN

1. Faces Dataset

![fake_41_faces](https://user-images.githubusercontent.com/26242097/52947263-38100700-339c-11e9-966a-f79e407f0909.png)

2. Stanford Cars Dataset

![Epoch 100](https://github.com/iArunava/DCGAN/blob/master/results/cars/fake_99.png)

3. Oxford Flowers Dataset

![Epoch 200](https://github.com/iArunava/DCGAN/blob/master/results/flowers/fake_99%20(2).png)

4. Simpsons Faces

![fake_99 (2)](https://user-images.githubusercontent.com/26242097/54181048-4f39a480-44c3-11e9-852a-8c1d944a943d.png)

<small> The images were resized to 64x64 before training </small>

## Pretrained models

1. Generator trained on the Stanford Cars Dataset for 100 epochs: [100 epoch trained Generator](http://bit.ly/g-100-cars)

2. Generator trained on the Oxford Flowers Dataset for 300 epochs: [300 epoch trained Generator](https://drive.google.com/file/d/1b-kGxNB4j2ummU9hUE959ew-EL2mYJP7/view?usp=drivesdk)

3. Generator trained on the Simpsons Faces Dataset for 300 epochs: [300 epoch trained Generator](https://www.dropbox.com/s/49sejsd1g5h6c4q/g-nm-99.pth?dl=1)

## License

The code in this repository is made available for free. Feel free to fork and go crazy.
Also, you are welcome to use this code in commercial purposes for free, with just proper linkage
redirecting back to this repository.
