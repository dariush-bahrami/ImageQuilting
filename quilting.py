from typing import Tuple

import numpy as np
from tqdm import tqdm


def estimate_quantity(
    target_size: Tuple[int, int],
    block_size: int,
    overlap: int,
) -> Tuple[int, int]:
    height, width = target_size
    n_h = np.ceil((height - block_size) / (block_size - overlap)).astype(int)
    n_w = np.ceil((width - block_size) / (block_size - overlap)).astype(int)
    return (n_h, n_w)


def get_patch_mse_error(
    texture_slice: np.ndarray,
    block_slice: np.ndarray,
) -> float:
    return np.square(texture_slice - block_slice).mean()


def update_error_and_get_min_index(
    mse_error: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # maintain min_index for 2nd row onwards and
    min_index = np.zeros((mse_error.shape[0] - 1, mse_error.shape[1]), dtype=int)
    errors = np.zeros_like(mse_error)
    errors[0] = mse_error[0]
    e = np.zeros((3, mse_error.shape[1]))
    e[0, 0] = np.inf
    e[-1, -1] = np.inf

    for i in range(1, mse_error.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        e[0, 1:] = errors[i - 1, :-1]
        e[1, :] = errors[i - 1]
        e[2, :-1] = errors[i - 1, 1:]

        # Get min_index
        min_array = e.min(0)
        min_arg = e.argmin(0) - 1
        min_index[i - 1, :] = min_arg
        # Set error_ij = e_ij + min_array
        error_ij = mse_error[i] + min_array
        errors[i] = error_ij
    return errors, min_index


def get_path(error: np.ndarray, min_index: np.ndarray) -> np.ndarray:
    # Check the last element and backtrack to find path
    path = []
    min_arg = np.argmin(error[-1])
    path.append(min_arg)

    # Backtrack to min path
    for idx in min_index[::-1]:
        min_arg = min_arg + idx[min_arg]
        path.append(min_arg)
    # Reverse to find full path
    path = path[::-1]
    return path


class Quilt:
    def __init__(
        self,
        texture_image: np.ndarray,
        block_size: int,
        overlap: int,
        tolerance: float,
        verbose=False,
    ):
        """An implementation of quilting algorithm for texture synthesis.

        Args:
            texture_image (np.ndarray): Image to quilt with values between 0 and 1.
            block_size (int): Size of each block in pixels.
            overlap (int): Number of pixels to overlap between blocks.
            tolerance (float): Error tolerance for the quilting algorithm.
            verbose (bool): Whether to print progress bar.
        """
        self.texture_image = texture_image
        self.block_size = block_size
        self.overlap = overlap
        self.tolerance = tolerance
        self.verbose = verbose
        self.error_matrix = self.initilize_error_matrix()

    def initilize_error_matrix(self):
        error_matrix_shape = (
            self.texture_image.shape[0] - self.block_size,
            self.texture_image.shape[1] - self.block_size,
        )
        error_matrix = np.full(error_matrix_shape, np.inf)
        return error_matrix

    def get_horizontal_patch_error(self, block: np.ndarray) -> np.ndarray:
        error_matrix = self.initilize_error_matrix()
        for i in range(self.texture_image.shape[0] - self.block_size):
            for j in range(self.texture_image.shape[1] - self.block_size):
                mse_error = get_patch_mse_error(
                    self.texture_image[i : i + self.block_size, j : j + self.overlap],
                    block[:, -self.overlap :],
                )
                if mse_error > 0:
                    error_matrix[i, j] = mse_error
        return error_matrix

    def get_vertical_patch_error(self, block: np.ndarray) -> np.ndarray:
        error_matrix = self.initilize_error_matrix()

        for i in range(self.texture_image.shape[0] - self.block_size):
            for j in range(self.texture_image.shape[1] - self.block_size):
                mse_error = get_patch_mse_error(
                    self.texture_image[i : i + self.overlap, j : j + self.block_size],
                    block[-self.overlap :, :],
                )
                if mse_error > 0:
                    error_matrix[i, j] = mse_error
        return error_matrix

    def get_patch_error(
        self,
        top_block: np.ndarray,
        left_block: np.ndarray,
    ) -> np.ndarray:
        error_matrix = self.initilize_error_matrix()
        for i in range(self.texture_image.shape[0] - self.block_size):
            for j in range(self.texture_image.shape[1] - self.block_size):
                mse_error = get_patch_mse_error(
                    self.texture_image[i : i + self.overlap, j : j + self.block_size],
                    top_block[-self.overlap :, :],
                )
                mse_error += get_patch_mse_error(
                    self.texture_image[i : i + self.block_size, j : j + self.overlap],
                    left_block[:, -self.overlap :],
                )

                if mse_error > 0:
                    error_matrix[i, j] = mse_error
        return error_matrix

    def get_patch_from_error_matrix(self, error_matrix: np.ndarray) -> np.ndarray:
        min_error = np.min(error_matrix)
        y, x = np.where(error_matrix < (1.0 + self.tolerance) * (min_error))
        if len(y) == 0:
            y = [np.random.randint(0, error_matrix.shape[0])]
            x = [np.random.randint(0, error_matrix.shape[1])]
        c = np.random.randint(len(y))
        y, x = y[c], x[c]
        return self.texture_image[y : y + self.block_size, x : x + self.block_size]

    def get_mask(self, path: list) -> np.ndarray:
        channels = self.texture_image.shape[2]
        mask = np.zeros((self.block_size, self.block_size, channels))
        for i in range(len(path)):
            mask[i, : path[i] + 1] = 1
        return mask

    def get_min_cut_horizontal_patch(
        self,
        block_1: np.ndarray,
        block_2: np.ndarray,
    ) -> np.ndarray:
        error = ((block_1[:, -self.overlap :] - block_2[:, : self.overlap]) ** 2).mean(
            2
        )
        error, min_index = update_error_and_get_min_index(error)
        path = get_path(error, min_index)
        mask = self.get_mask(path)
        result_block = np.zeros(block_1.shape)
        result_block[:, : self.overlap] = block_1[:, -self.overlap :]
        result_block = result_block * mask + block_2 * (1 - mask)
        return result_block

    def get_min_cut_vertical_patch(
        self,
        block_1: np.ndarray,
        block_2: np.ndarray,
    ) -> np.ndarray:
        result_block = self.get_min_cut_horizontal_patch(
            np.rot90(block_1),
            np.rot90(block_2),
        )
        return np.rot90(result_block, 3)

    def get_min_cut_patch(
        self,
        left_block: np.ndarray,
        top_block: np.ndarray,
        patch_block: np.ndarray,
    ) -> np.ndarray:
        error = (
            (left_block[:, -self.overlap :] - patch_block[:, : self.overlap]) ** 2
        ).mean(2)
        error, min_index = update_error_and_get_min_index(error)
        path = get_path(error, min_index)
        mask1 = self.get_mask(path)

        ###################################################################
        ## Now for vertical one
        error = (
            (
                np.rot90(top_block)[:, -self.overlap :]
                - np.rot90(patch_block)[:, : self.overlap]
            )
            ** 2
        ).mean(2)
        error, min_index = update_error_and_get_min_index(error)
        path = get_path(error, min_index)
        mask2 = self.get_mask(path)
        mask2 = np.rot90(mask2, 3)

        mask2[: self.overlap, : self.overlap] = np.maximum(
            mask2[: self.overlap, : self.overlap]
            - mask1[: self.overlap, : self.overlap],
            0,
        )

        # Put first mask
        result_block = np.zeros(patch_block.shape)
        result_block[:, : self.overlap] = (
            mask1[:, : self.overlap] * left_block[:, -self.overlap :]
        )
        result_block[: self.overlap, :] = (
            result_block[: self.overlap, :]
            + mask2[: self.overlap, :] * top_block[-self.overlap :, :]
        )
        result_block = result_block + (1 - np.maximum(mask1, mask2)) * patch_block
        return result_block

    def fill_first_row(self, result: np.ndarray):
        start = self.block_size - self.overlap
        stop = result.shape[1] - self.overlap
        step = self.block_size - self.overlap
        for block_index in range(start, stop, step):
            # Find horizontal error for this block
            # Calculate min, find index having tolerance
            # Choose one randomly among them
            # block_index = block index to put in
            block = result[
                : self.block_size,
                (block_index - self.block_size + self.overlap) : (
                    block_index + self.overlap
                ),
            ]
            error_matrix = self.get_horizontal_patch_error(block)
            patch_block = self.get_patch_from_error_matrix(error_matrix)
            min_cut_patch = self.get_min_cut_horizontal_patch(block, patch_block)
            result[: self.block_size, block_index : (block_index + self.block_size)] = (
                min_cut_patch
            )
        return result

    def fill_first_column(self, result: np.ndarray):
        start = self.block_size - self.overlap
        stop = result.shape[0] - self.overlap
        step = self.block_size - self.overlap
        for block_index in range(start, stop, step):
            # Find vertical error for this block
            # Calculate min, find index having tolerance
            # Choose one randomly among them
            # block_index = block index to put in
            block = result[
                (block_index - self.block_size + self.overlap) : (
                    block_index + self.overlap
                ),
                : self.block_size,
            ]
            error_matrix = self.get_vertical_patch_error(block)
            patch_block = self.get_patch_from_error_matrix(error_matrix)
            min_cut_patch = self.get_min_cut_vertical_patch(block, patch_block)

            result[block_index : (block_index + self.block_size), : self.block_size] = (
                min_cut_patch
            )
        return result

    def fill_rows_and_columns(self, result: np.ndarray, quantity: Tuple[int, int]):
        if self.verbose is True:
            pbar = tqdm(total=quantity[0] * quantity[1], desc="Synthesizing")
        n_h, n_w = quantity
        for i in range(1, n_h + 1):
            for j in range(1, n_w + 1):
                # Choose the starting index for the texture_image placement
                block_index_i = i * (self.block_size - self.overlap)
                block_index_j = j * (self.block_size - self.overlap)
                # Find the left and top block, and the min errors independently
                left_block = result[
                    (block_index_i) : (block_index_i + self.block_size),
                    (block_index_j - self.block_size + self.overlap) : (
                        block_index_j + self.overlap
                    ),
                ]
                top_block = result[
                    (block_index_i - self.block_size + self.overlap) : (
                        block_index_i + self.overlap
                    ),
                    (block_index_j) : (block_index_j + self.block_size),
                ]

                error_matrix = self.get_patch_error(top_block, left_block)
                patch_block = self.get_patch_from_error_matrix(error_matrix)
                min_cut_patch = self.get_min_cut_patch(
                    left_block,
                    top_block,
                    patch_block,
                )

                result[
                    (block_index_i) : (block_index_i + self.block_size),
                    (block_index_j) : (block_index_j + self.block_size),
                ] = min_cut_patch
                if self.verbose is True:
                    pbar.update(1)
        return result

    def __call__(self, quantity: Tuple[int, int]) -> np.ndarray:
        """Synthesize texture.

        Args:
            quantity (tuple[int, int]): Number of textures in (height, width).

        Returns:
            np.ndarray: Quilted texture.
        """

        n_h, n_w = quantity
        result_height = self.block_size + n_h * (self.block_size - self.overlap)
        result_width = self.block_size + n_w * (self.block_size - self.overlap)
        result_channels = self.texture_image.shape[2]
        result = np.zeros((result_height, result_width, result_channels))

        # Starting index and block
        rand_h = np.random.randint(self.texture_image.shape[0] - self.block_size)
        rand_w = np.random.randint(self.texture_image.shape[1] - self.block_size)

        initial_block = self.texture_image[
            rand_h : rand_h + self.block_size, rand_w : rand_w + self.block_size
        ]
        result[: self.block_size, : self.block_size, :] = initial_block

        # Fill the first row
        if self.verbose is True:
            print("Filling first row...")
        result = self.fill_first_row(result)

        ### Fill the first column
        if self.verbose is True:
            print("Filling first column...")
        result = self.fill_first_column(result)

        ### Fill in the other rows and columns
        result = self.fill_rows_and_columns(result, quantity)
        return result


def get_quilted_image(
    texture: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    overlap: int,
    tolerance: float = 0.1,
    verbose: bool = False,
) -> np.ndarray:
    quilter = Quilt(texture / 255.0, block_size, overlap, tolerance, verbose)
    quantity = estimate_quantity(
        (height, width),
        block_size,
        overlap,
    )
    texture_canvas = quilter(quantity)
    texture_canvas = (texture_canvas * 255.0).astype(np.uint8)
    return texture_canvas
