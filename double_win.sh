# echo vgg8 double
# python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s vgg8_double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo vgg8
python train_student.py --path_t ./save/student_model/S~vgg8_double_T~vgg13_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/vgg8_double_best.pth \
--epoch 1 --distill kd --model_s vgg8 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo vgg13 double
# python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s vgg13_double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo vgg13
python train_student.py --path_t ./save/student_model/S~vgg13_double_T~vgg13_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/vgg13_double_best.pth \
--epoch 1 --distill kd --model_s vgg13 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo mobilev2 double
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s MobileNetV2Double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo mobilev2
python train_student.py --path_t ./save/student_model/S~MobileNetV2Double_T~resnet32x4_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/MobileNetV2Double_best.pth \
--epoch 1 --distill kd --model_s MobileNetV2 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo shufflev1 double
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s ShuffleV1Double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo shufflev1
python train_student.py --path_t ./save/student_model/S~ShuffleV1Double_T~resnet32x4_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/ShuffleV1Double_best.pth \
--epoch 1 --distill kd --model_s ShuffleV1 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo shufflev2 double
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s ShuffleV2Double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo shufflev2
python train_student.py --path_t ./save/student_model/S~ShuffleV2Double_T~resnet32x4_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/ShuffleV2Double_best.pth \
--epoch 1 --distill kd --model_s ShuffleV2 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo wrn_16_2 double
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s wrn_16_2_double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo wrn_16_2
python train_student.py --path_t ./save/student_model/S~wrn_16_2_double_T~resnet32x4_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/wrn_16_2_double_best.pth \
--epoch 1 --distill kd --model_s wrn_16_2 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128

# echo resnet8x4 double
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 1 --distill kd --model_s resnet8x4_double -r 1 -a 1 -b 1 --trial 0 \
# --hkd_initial_weight 100 --hkd_decay 0.7 \
# --batch_size 128

echo resnet8x4
python train_student.py --path_t ./save/student_model/S~resnet8x4_double_T~resnet32x4_cifar100_hkd_r~1.0_a~1.0_b~1.0_0/resnet8x4_double_best.pth \
--epoch 1 --distill kd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 0 \
--hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128