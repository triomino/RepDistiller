python train_student.py --path_t ./save/student_model/S:vgg8_double_T:vgg13_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/vgg8_double_best.pth  \
--distill hkd --model_s vgg8 -r 1 -a 1 -b 1 --trial 1

python train_student.py --path_t ./save/student_model/S:MobileNetV2Double_T:wrn_40_2_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/MobileNetV2Double_best.pth  \
--distill hkd --model_s MobileNetV2 -r 1 -a 1 -b 1 --trial 1 --epoch 1

python train_student.py --path_t ./save/student_model/S:resnet8x4_double_T:resnet32x4_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/resnet8x4_double_best.pth  \
--distill hkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 1

python train_student.py --path_t ./save/student_model/S:ShuffleV1Double_T:resnet32x4_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/ShuffleV1Double_best.pth  \
--distill hkd --model_s ShuffleV1 -r 1 -a 1 -b 1 --trial 1

python train_student.py --path_t ./save/student_model/S:ShuffleV2Double_T:vgg13_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/ShuffleV2Double_best.pth  \
--distill hkd --model_s ShuffleV2 -r 1 -a 1 -b 1 --trial 1

python train_student.py --path_t ./save/student_model/S:vgg13_double_T:resnet32x4_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/vgg13_double_best.pth  \
--distill hkd --model_s vgg13 -r 1 -a 1 -b 1 --trial 1

python train_student.py --path_t ./save/student_model/S:wrn_16_2_double_T:wrn_40_2_cifar100_pkt_r:1.0_a:1.0_b:30000.0_1/wrn_16_2_double_best.pth  \
--distill hkd --model_s wrn_16_2 -r 1 -a 1 -b 1 --trial 1