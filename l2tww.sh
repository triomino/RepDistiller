python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  \
--distill l2tww --model_s resnet32 -r 1 -a 1 -b 1 --trial test