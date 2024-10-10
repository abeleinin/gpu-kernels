import os
import sys

from PIL import Image

import numpy as np
import mlx.core as mx


# converts (r,g,b) color values to grey scale equivalent
def luminance(r: float, g: float, b: float):
    return r*0.21 + g*0.72 + b*0.07


# metal kernel using MLX custom kernels to convert a given 
# (H,W,C) uint8 color image to a greyscale (H,W,C) equivalent
def greyscale_kernel(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * \
                 threads_per_threadgroup.x + \
                 thread_position_in_threadgroup.x;
        uint j = threadgroup_position_in_grid.y * \
                 threads_per_threadgroup.y + \
                 thread_position_in_threadgroup.y;

        if (i < a_shape[0] && j < a_shape[1]) {
            uint index = i * a_shape[1] + j;
            uint rgbOffset = index*3;
            float r = a[rgbOffset];
            float g = a[rgbOffset+1];
            float b = a[rgbOffset+2];
            float grey = r*0.21 + g*0.72 + b*0.07;
            out[rgbOffset] = grey;
            out[rgbOffset+1] = grey;
            out[rgbOffset+2] = grey;
        }
    """

    kernel = mx.fast.metal_kernel(
        name="greyscale",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    output = kernel(
        inputs=[a],
        grid=(a.shape[0], a.shape[1], 1),
        threadgroup=(8, 8, 1),
        output_shapes=[a.shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
        init_value=0,
    )

    return output[0]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: greyscale_metal.py path/to/img.jpg")
        sys.exit(1)

    # open image
    img_path = sys.argv[1]
    img = Image.open(img_path)
    img = mx.array(np.array(img))

    # call metal kernel
    grey_img = greyscale_kernel(img)

    # convert back to uint8 PIL.Image and save as jpg
    grey_img_uint8 = (np.array(grey_img)).astype('uint8')
    image = Image.fromarray(grey_img_uint8)
    input_name, _ = os.path.splitext(os.path.basename(img_path))
    image.save(f'grey_{input_name}.jpg', format='JPEG')

