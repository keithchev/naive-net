import numpy as np

def tile_filters(filter_stack_):

    filter_stack = filter_stack_.copy()

    filter_stack -= filter_stack.min()
    filter_stack /= filter_stack.max()

    F, C, H, W = filter_stack.shape

    filter_stack_ = np.zeros((F, C, H + 1, W + 1))

    filter_stack_[:,:, :H, :W] = filter_stack
    filter_stack = filter_stack_

    F, C, H, W = filter_stack.shape

    n_rows = int(np.ceil(np.sqrt(F)))
    n_cols = int(np.ceil(1.0*F/n_rows))

    if C != 3:
        print('Input depth not 3; making grayscale tiles')
        tile = np.zeros((n_rows*H, n_cols*w, 1), dtype=filter_stack.dtype)
        channels = 0
    else:
        channels = [0,1,2]
        tile = np.zeros((n_rows * H, n_cols * W, 3), dtype=filter_stack.dtype)

    ii = 0
    for rr in np.arange(n_rows):
        for cc in np.arange(n_cols):

            filt = filter_stack[ii, channels, :, :].transpose(1, 2, 0)

            tile[(rr*H):((rr+1)*H), (cc*W):((cc+1)*W), :] = filt
            ii += 1

            if ii >= F:
                break

    return tile