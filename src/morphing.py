#######################################################
#
#   Automatic Image Morphing
#   András Jankovics andras@jankovics.net , 2020, adapted from
#
#   https://github.com/ddowd97/Morphing
#   Author:     David Dowd
#   Email:      ddowd97@gmail.com
#
#   Additional features :
#    - automatic triangle points selection by cv2.goodFeaturesToTrack()
#    - No GUI, single file program
#    - batch processing: transition between many images, not just 2
#    - optional subpixel processing to fix image artifacts
#    - automatic image dimensions safety (the dimensions of the first image defines the output)
#
#   Install dependencies:
#    pip install scipy numpy matplotlib opencv-python
#
#   Recommended postprocessing:
#    Install FFmpeg https://ffmpeg.org/
#    Example from command line:
#     ffmpeg -framerate 15 -i image%d.png output.avi
#     ffmpeg -framerate 15 -i image%d.png output.gif
#
#   TODOs:
#    - testing, error checks, sanity checks
#    - speed optimization in interpolatePoints()
#    - RGBA support, currently it's only RGB
#    - tuning the parameters of cv2.goodFeaturesToTrack() in autofeaturepoints() / giving user control
#    - built-in video output with cv2 ?
#    - image scaling uses cv2.INTER_CUBIC ; tuning / giving user control ?
#    - LinAlgError ? Image dimensions should be even numbers?
#       related: https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
#         File "batchautomorph.py", line 151, in interpolatePoints
#           righth = np.linalg.solve(tempRightMatrix, targetVertices)
#         File "<__array_function__ internals>", line 5, in solve
#         File "numpy\linalg\linalg.py",
#           line 403, in solve
#           r = gufunc(a, b, signature=signature, extobj=extobj)
#         File "numpy\linalg\linalg.py",
#           line 97, in _raise_linalgerror_singular
#           raise LinAlgError("Singular matrix")
#           numpy.linalg.LinAlgError: Singular matrix
#
#######################################################

from typing import List, Tuple
import cv2, time, argparse, ast
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path
import numpy as np
import os

#######################################################
#   https://github.com/ddowd97/Morphing
#   Author:     David Dowd
#   Email:      ddowd97@gmail.com
#######################################################


class Triangle:
    def __init__(self, vertices: np.ndarray) -> None:
        if not isinstance(vertices, np.ndarray):
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError("Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")

        self.vertices: np.ndarray = vertices
        self.minX: int = int(self.vertices[:, 0].min())
        self.maxX: int = int(self.vertices[:, 0].max())
        self.minY: int = int(self.vertices[:, 1].min())
        self.maxY: int = int(self.vertices[:, 1].max())

    def get_points(self) -> np.ndarray:
        x_list = range(self.minX, self.maxX + 1)
        y_list = range(self.minY, self.maxY + 1)
        empty_list = list((x, y) for x in x_list for y in y_list)

        points = np.array(empty_list, np.float64)
        p = Path(self.vertices)
        grid = p.contains_points(points)
        mask = grid.reshape(self.maxX - self.minX + 1, self.maxY - self.minY + 1)

        true_array = np.where(mask)
        coord_array = np.vstack(
            (
                true_array[0] + self.minX,
                true_array[1] + self.minY,
                np.ones(true_array[0].shape[0]),
            )
        )

        return coord_array


def load_triangles(
    limg: np.ndarray, rimg: np.ndarray, feature_grid_size: int, show_features: bool
) -> Tuple[List[Triangle], List[Triangle]]:
    left_tri_list: List[Triangle] = []
    right_tri_list: List[Triangle] = []

    lrlists = auto_feature_points(limg, rimg, feature_grid_size, show_features)

    left_array = np.array(lrlists[0], np.float64)
    right_array = np.array(lrlists[1], np.float64)
    delaunay_tri = Delaunay(left_array)

    left_np = left_array[delaunay_tri.simplices]
    right_np = right_array[delaunay_tri.simplices]

    for x, y in zip(left_np, right_np):
        left_tri_list.append(Triangle(x))
        right_tri_list.append(Triangle(y))

    return left_tri_list, right_tri_list


class Morpher:
    def __init__(
        self,
        left_image: np.ndarray,
        left_triangles: List[Triangle],
        right_image: np.ndarray,
        right_triangles: List[Triangle],
    ) -> None:
        self.left_image = np.ndarray.copy(left_image)
        self.left_triangles = left_triangles
        self.right_image = np.ndarray.copy(right_image)
        self.right_triangles = right_triangles

        self.left_interpolation = RectBivariateSpline(
            np.arange(self.left_image.shape[0]),
            np.arange(self.left_image.shape[1]),
            self.left_image,
        )
        self.right_interpolation = RectBivariateSpline(
            np.arange(self.right_image.shape[0]),
            np.arange(self.right_image.shape[1]),
            self.right_image,
        )

    def get_image_at_alpha(self, alpha: float, smooth_mode: int) -> np.ndarray:
        for left_triangle, right_triangle in zip(
            self.left_triangles, self.right_triangles
        ):
            self.interpolate_points(left_triangle, right_triangle, alpha)

        blend_arr = (1 - alpha) * self.left_image + alpha * self.right_image
        blend_arr = blend_arr.astype(np.uint8)
        return blend_arr

    def interpolate_points(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(
            leftTriangle.vertices
            + (rightTriangle.vertices - leftTriangle.vertices) * alpha
        )
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array(
            [
                [leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                [0, 0, 0, leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1],
                [leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1, 0, 0, 0],
                [0, 0, 0, leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1],
                [leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1, 0, 0, 0],
                [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1],
            ]
        )
        tempRightMatrix = np.array(
            [
                [
                    rightTriangle.vertices[0][0],
                    rightTriangle.vertices[0][1],
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    rightTriangle.vertices[0][0],
                    rightTriangle.vertices[0][1],
                    1,
                ],
                [
                    rightTriangle.vertices[1][0],
                    rightTriangle.vertices[1][1],
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    rightTriangle.vertices[1][0],
                    rightTriangle.vertices[1][1],
                    1,
                ],
                [
                    rightTriangle.vertices[2][0],
                    rightTriangle.vertices[2][1],
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    rightTriangle.vertices[2][0],
                    rightTriangle.vertices[2][1],
                    1,
                ],
            ]
        )
        lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
        righth = np.linalg.solve(tempRightMatrix, targetVertices)
        leftH = np.array(
            [
                [lefth[0][0], lefth[1][0], lefth[2][0]],
                [lefth[3][0], lefth[4][0], lefth[5][0]],
                [0, 0, 1],
            ]
        )
        rightH = np.array(
            [
                [righth[0][0], righth[1][0], righth[2][0]],
                [righth[3][0], righth[4][0], righth[5][0]],
                [0, 0, 1],
            ]
        )
        leftinvH = np.linalg.inv(leftH)
        rightinvH = np.linalg.inv(rightH)
        targetPoints = targetTriangle.get_points()  # TODO: ~ 17-18% of runtime

        leftSourcePoints = np.transpose(np.matmul(leftinvH, targetPoints))
        rightSourcePoints = np.transpose(np.matmul(rightinvH, targetPoints))
        targetPoints = np.transpose(targetPoints)

        for x, y, z in zip(
            targetPoints, leftSourcePoints, rightSourcePoints
        ):  # TODO: ~ 53% of runtime
            self.left_image[int(x[1])][int(x[0])] = self.left_interpolation(y[1], y[0])
            self.right_image[int(x[1])][int(x[0])] = self.right_interpolation(
                z[1], z[0]
            )


########################################################################################################


# Automatic feature points
def auto_feature_points(leimg, riimg, featuregridsize, showfeatures):
    result = [[], []]
    for idx, img in enumerate([leimg, riimg]):
        try:
            if showfeatures:
                print(img.shape)

            # add the 4 corners to result
            result[idx] = [
                [0, 0],
                [(img.shape[1] - 1), 0],
                [0, (img.shape[0] - 1)],
                [(img.shape[1] - 1), (img.shape[0] - 1)],
            ]

            h = int(img.shape[0] / featuregridsize) - 1
            w = int(img.shape[1] / featuregridsize) - 1

            for i in range(0, featuregridsize):
                for j in range(0, featuregridsize):
                    # crop to a small part of the image and find 1 feature pont or middle point
                    crop_img = img[(j * h) : (j * h) + h, (i * w) : (i * w) + w]
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    featurepoints = cv2.goodFeaturesToTrack(
                        gray, 1, 0.1, 10
                    )  # TODO: parameters can be tuned
                    if featurepoints is None:
                        featurepoints = [[[h / 2, w / 2]]]
                    featurepoints = np.int0(featurepoints)

                    # add feature point to result, optionally draw
                    for featurepoint in featurepoints:
                        x, y = featurepoint.ravel()
                        y = y + (j * h)
                        x = x + (i * w)
                        if showfeatures:
                            cv2.circle(img, (x, y), 3, 255, -1)
                        result[idx].append([x, y])

            # optionally draw features
            if showfeatures:
                cv2.imshow("", img)
                cv2.waitKey(0)

        except Exception as ex:
            print(ex)
    return result


#####


def initmorph(startimgpath, endimgpath, featuregridsize, subpixel, showfeatures, scale):
    timerstart = time.time()

    # left image load
    leftImageRaw = cv2.imread(startimgpath)
    # scale image if custom scaling
    if scale != 1.0:
        leftImageRaw = cv2.resize(
            leftImageRaw,
            (int(leftImageRaw.shape[1] * scale), int(leftImageRaw.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    # upscale image if subpixel calculation is enabled
    if subpixel > 1:
        leftImageRaw = cv2.resize(
            leftImageRaw,
            (leftImageRaw.shape[1] * subpixel, leftImageRaw.shape[0] * subpixel),
            interpolation=cv2.INTER_CUBIC,
        )

    # right image load
    rightImageRaw = cv2.imread(endimgpath)
    # resize image
    rightImageRaw = cv2.resize(
        rightImageRaw,
        (leftImageRaw.shape[1], leftImageRaw.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    leftImageARR = np.asarray(leftImageRaw)
    rightImageARR = np.asarray(rightImageRaw)

    # autofeaturepoints() is called in loadTriangles()
    triangleTuple = load_triangles(
        leftImageRaw, rightImageRaw, featuregridsize, showfeatures
    )

    # Morpher objects for color layers BGR
    morphers = [
        Morpher(
            leftImageARR[:, :, 0],
            triangleTuple[0],
            rightImageARR[:, :, 0],
            triangleTuple[1],
        ),
        Morpher(
            leftImageARR[:, :, 1],
            triangleTuple[0],
            rightImageARR[:, :, 1],
            triangleTuple[1],
        ),
        Morpher(
            leftImageARR[:, :, 2],
            triangleTuple[0],
            rightImageARR[:, :, 2],
            triangleTuple[1],
        ),
    ]

    print(
        "\r\nSubsequence init time: "
        + "{0:.2f}".format(time.time() - timerstart)
        + " s "
    )

    return morphers


####


def save_morphed_frame(image: np.ndarray, frame_count: int, output_prefix: str) -> None:
    # Create directory if it does not exist
    directory = os.path.dirname(output_prefix)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Create the filename
    filename = f"{output_prefix}{frame_count}.png"

    # Save the image
    cv2.imwrite(filename, image)

    # Log the operation
    print(f"{filename} saved, dimensions {image.shape}")


def apply_median_filter(image: np.ndarray, smoothing: int) -> np.ndarray:
    return np.array(median_filter(image, smoothing))


def generate_morphed_frame(
    morphers: List[Morpher], alpha: float, smoothing: int
) -> np.ndarray:
    channels = [morpher.get_image_at_alpha(alpha, True) for morpher in morphers]

    if smoothing > 0:
        channels = [apply_median_filter(channel, smoothing) for channel in channels]

    return np.dstack(channels)


def morph_process(
    morphers: List[Morpher],
    frame_rate: int,
    output_prefix: str,
    subpixel: int,
    smoothing: int,
) -> None:
    frame_count = 1

    for i in range(1, frame_rate):
        timer_start = time.time()
        alpha = i / frame_rate

        morphed_frame = generate_morphed_frame(morphers, alpha, smoothing)

        if subpixel > 1:
            morphed_frame = cv2.resize(
                morphed_frame,
                (
                    int(morphed_frame.shape[1] / subpixel),
                    int(morphed_frame.shape[0] / subpixel),
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        save_morphed_frame(morphed_frame, frame_count, output_prefix)
        frame_count += 1

        timer_elapsed = time.time() - timer_start
        us_per_pixel = (
            1_000_000
            * timer_elapsed
            / (morphed_frame.shape[0] * morphed_frame.shape[1])
        )
        print(f"Time: {timer_elapsed:.2f} s; μs/pixel: {us_per_pixel:.2f}")


####


def batchmorph(
    imgs: List[str],
    featuregridsize: int,
    subpixel: int,
    showfeatures: bool,
    framerate: int,
    outimgprefix: str,
    smoothing: int,
    scale: float,
) -> None:
    totaltimerstart = time.time()

    for idx, (img1, img2) in enumerate(zip(imgs[:-1], imgs[1:])):
        morph_params = initmorph(
            img1, img2, featuregridsize, subpixel, showfeatures, scale
        )
        morph_process(morph_params, framerate, outimgprefix, subpixel, smoothing)

    total_elapsed_time = time.time() - totaltimerstart
    print(f"\r\nDone. Total time: {total_elapsed_time:.2f} s")


def generate_morphing_between_images(
    path_to_images: List[str],
):
    batchmorph(
        imgs=path_to_images,
        featuregridsize=20,
        subpixel=1,
        showfeatures=False,
        framerate=10,
        outimgprefix="morphing_output/morphed_",
        smoothing=0,
        scale=1.0,
    )


if __name__ == "__main__":
    generate_morphing_between_images(
        ["generated_image_0.png", "generated_image_1.png", "generated_image1.png"]
    )
