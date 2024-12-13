#torchrun --nnodes=1 --nproc_per_node=2 -m applications.dc_ae.generate_reference dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=test fid.save_path=assets/data/fid/imagenet_512_val.npz
#torchrun --nnodes=1 --nproc_per_node=2 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=dc-ae-f32c32-in-1.0 run_dir=tmp

# Expected results:
#   fid: 0.2167766520628902
#   psnr: 26.1489275
#   ssim: 0.710486114025116
#   lpips: 0.0802311897277832