import mlx.core as mx

def matmul_kernel(a: mx.array, b: mx.array):
    source = """
        uint3 gid = threadgroup_position_in_grid;

        if (gid.x >= a_shape[0] || gid.y >= a_shape[1]) {
            return;
        }

        float sum = 0.0;
        for (uint k = 0; k < a_shape[0]; ++k) {
            sum += a[gid.x * a_shape[0] + k] * b[k * a_shape[1] + gid.y];
        }
    
        out[gid.x * a_shape[1] + gid.y] = sum;
    """

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
    )

    output = kernel(
        inputs=[a, b],
        grid=(a.shape[0], a.shape[1], 1),
        threadgroup=(1, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
        init_value=0,
    )

    return output[0]


if __name__ == '__main__':
    a = mx.arange(9, dtype=mx.float32).reshape(3,3)
    b = mx.arange(9, dtype=mx.float32).reshape(3,3).transpose()

    res = matmul_kernel(a, b)

    assert(mx.allclose(res, a @ b).item())
