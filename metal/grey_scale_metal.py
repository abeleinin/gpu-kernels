import mlx.core as mx
import PIL.Image
import numpy as np


def luminance(r, g, b):
    return r*0.21 + g*0.72 + b*0.07


def grey_scale_kernel(a: mx.array):
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
            float r = float(a[rgbOffset]);
            float g = float(a[rgbOffset+1]);
            float b = float(a[rgbOffset+2]);
            float grey = r * 0.21 + g * 0.72 + b * 0.07;
            out[rgbOffset] = grey;
            out[rgbOffset+1] = grey;
            out[rgbOffset+2] = grey;
        }
    """

    kernel = mx.fast.metal_kernel(
        name="grey_scale",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    output = kernel(
        inputs=[a],
        grid=(256, 256, 1),
        threadgroup=(16, 16, 1),
        output_shapes=[a.shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
        init_value=0,
    )

    return output[0]


if __name__ == '__main__':
    img = PIL.Image.open('pfp.jpeg')
    img = mx.array(np.array(img))
    grey_img = grey_scale_kernel(img)

    grey_img_uint8 = (np.array(grey_img)/ 256).astype('uint8')
    image = PIL.Image.fromarray(grey_img_uint8)
    image.save('grey_pfp.jpg', format='JPEG')
