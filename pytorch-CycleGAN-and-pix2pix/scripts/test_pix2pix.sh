python test.py --dataroot ./datasets/$1 --name '$1'_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch --gpu_ids $2
