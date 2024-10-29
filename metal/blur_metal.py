import os
import sys

from PIL import Image

import numpy as np
import mlx.core as mx


# metal kernel using MLX custom kernels to apply a mean blur
# to the given (H,W,C) uint8 color image 
def blur_kernel(a: mx.array):
    header = """
        constant int BLUR_SIZE = 3;
        constant uint CHANNELS = 3;
    """

    source = """
        uint col = threadgroup_position_in_grid.x * \
                   threads_per_threadgroup.x + \
                   thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y * \
                   threads_per_threadgroup.y + \
                   thread_position_in_threadgroup.y;

        if (col < a_shape[0] && row < a_shape[1]) {
            int pixValR = 0;
            int pixValG = 0;
            int pixValB = 0;

            int pixels = 0;

            for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
                for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                    uint currRow = row + blurRow;
                    uint currCol = col + blurCol;

                    if (currRow >= 0 && currRow < a_shape[1] && currCol >= 0 && currCol < a_shape[0]) {
                        uint offset = currRow*a_shape[0] + currCol;
                        uint rgbOffset = offset*CHANNELS;
                        pixValR += a[rgbOffset    ];
                        pixValG += a[rgbOffset + 1];
                        pixValB += a[rgbOffset + 2];
                        ++pixels;
                    }
                }
            }

            uint offset = (row*a_shape[0] + col) * CHANNELS;
            out[offset    ] = (uint)(pixValR/pixels);
            out[offset + 1] = (uint)(pixValG/pixels);
            out[offset + 2] = (uint)(pixValB/pixels);
        }
    """

    kernel = mx.fast.metal_kernel(
        name="blur",
        input_names=["a"],
        output_names=["out"],
        header=header,
        source=source,
    )

    output = kernel(
        inputs=[a],
        grid=(a.shape[0], a.shape[1], 1),
        threadgroup=(8, 8, 1),
        output_shapes=[a.shape],
        output_dtypes=[mx.uint8],
        stream=mx.gpu,
        init_value=0
    )

    return output[0]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: blur_metal.py path/to/img.jpg")
        sys.exit(1)

    # open image
    img_path = sys.argv[1]
    img = Image.open(img_path)
    img = mx.array(np.array(img))

    # call metal kernel
    blur_img = blur_kernel(img)

    # convert into PIL.Image and save as jpg
    image = Image.fromarray(np.array(blur_img))
    input_name, _ = os.path.splitext(os.path.basename(img_path))
    image.save(f'blur_{input_name}.jpg', format='JPEG')
