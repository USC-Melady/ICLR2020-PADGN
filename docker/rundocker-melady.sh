docker run -it --rm --gpus all \
  --shm-size 16G \
  --name physics-informed-GN \
  -u $(id -u):$(id -g) \
  -v ~/physics-informed-GN:/workspace/physics-informed-GN \
  -p 127.0.0.1:19954:19954 -p 127.0.0.1:19900-19950:19900-19950 -p 127.0.0.1:18940:22 \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  mengcz/physics-informed-gn
