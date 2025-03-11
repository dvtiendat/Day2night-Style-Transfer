# Day2Night Style Transfer

Day2Night CycleGAN model to effectively transfer day images of traffic camera surveillance to night images and vice-versa. This model can be used for data augmentation to enhance the quantity of images in object detection dataset.

## Example 
Day to Night transformation

![image](https://github.com/user-attachments/assets/e7d668fd-4e97-42fa-9340-e3b4dace493d)

Night to Day transformation 

![image](https://github.com/user-attachments/assets/bb3faf79-7799-4813-95eb-cbdec7a17bc7)

Each generation take approximately 0.005s.
## Usage
Training
```
bash ./train.sh
```
or
```
python train.py --batch_size --num_epochs  --root_day --root_night --load_checkpoint
```
Configurations are editable as listed in train.py
