if [ "$1" = "cars" ]; then
    wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz -O ./datasets/car.tgz
    tar -xzf car.tgz
    mkdir cars_train/car/
    mv cars_train/* cars_train/car/

elif [ "$1" = "flowers" ]; then
    wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -O ./datasets/flowers.tgz
    tar -xzf ./datasets/flowers.tgz
    mkdir ./datasets/flowers/
    mv ./datasets/jpg ./datasets/flowers/

elif [ "$1" = "dogs" ]; then
    wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O ./datasets/dogs.tar
    tar -xf dogs.tar
else
    echo "Invalid Argument"
fi
