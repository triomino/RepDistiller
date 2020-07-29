echo vgg8
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s vgg8_double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo vgg13
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s vgg13_double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo mobilev2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s MobileNetV2Double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo shufflev1
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s ShuffleV1Double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo shufflev2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s ShuffleV2Double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo wrn_16_2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s wrn_16_2_double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

echo resnet8x4
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 1 --distill hkd --model_s resnet8x4_double -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128