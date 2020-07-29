# teacher -> aux
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
# --epoch 240 --distill hkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 0 \
# --teacher_layer 3 --student_layer 3 --hkd_initial_weight 1 --hkd_decay 1 \
# --batch_size 128

# aux -> student
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--epoch 240 --distill hkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 0 \
--teacher_layer 0 1 2 3 --student_layer 0 1 2 3 --hkd_initial_weight 100 --hkd_decay 0.7 \
--batch_size 128