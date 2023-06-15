# Version 5.0
import cv2
import numpy as np
import pyopencl as cl
from PIL import Image


def generate_gradient_palette(stops, num_intermediate_colors):
    """
    Generates a gradient palette based on defined stops.

    Args:
        stops: A list of color stops, each represented as an RGB tuple.
        num_intermediate_colors: The number of intermediate colors to generate between each stop.

    Returns:
        A numpy array representing the gradient palette.
    """
    palette = []
    for i in range(len(stops) - 1):
        color1 = np.array(stops[i], dtype=np.uint8)
        color2 = np.array(stops[i + 1], dtype=np.uint8)
        for j in range(num_intermediate_colors + 1):
            t = j / float(num_intermediate_colors)
            interpolated_color = (1 - t) * color1 + t * color2
            palette.append(interpolated_color)
    return np.array(palette, dtype=np.uint8)


def map_color_palette(image, palette):
    """
    Maps a user-defined color palette based on the nearest color in the input image.

    Args:
        image: A numpy array representing the input image (gray-scale, or red-channel only).
        palette: A numpy array representing the colors in the palette.

    Returns:
        A numpy array representing the output image (RGB).
    """

    # Create a context and a command queue.
    # platform = cl.get_platforms()[0]
    # device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    context = cl.create_some_context()
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Define the kernel code as a string
    kernel_code = """
__kernel void map_color_palette(
    __global const uchar *input,
    __global uchar *output,
    __global const uchar *palette,
    int palette_size,
    int width,
    int height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = (y * width + x)*3;
    int input_value = input[index+1];

    int palette_index = (int) (((float)input_value/255.0)*palette_size);
    
    uchar red = palette[palette_index*3];
    uchar green = palette[palette_index*3+1];
    uchar blue = palette[palette_index*3+2];

    output[index] = blue;
    output[index + 1] = green;
    output[index+2] = red;
}
"""

    # Compile the kernel code
    program = cl.Program(context, kernel_code).build()

    # Create buffers for the input image, output image, and palette.
    image_buffer = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, size=image.nbytes)
    palette_buffer = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=palette)

    # Set the arguments for the kernel
    kernel = program.map_color_palette
    kernel.set_args(image_buffer, output_buffer, palette_buffer, np.int32(len(palette)), np.int32(image.shape[1]),
                    np.int32(image.shape[0]))

    # Define the work size and execute the kernel
    work_size = (image.shape[1], image.shape[0])
    cl.enqueue_nd_range_kernel(queue, kernel, work_size, None)

    # Read the output image from the buffer.
    output = np.empty_like(image)
    cl._enqueue_read_buffer(queue, output_buffer, output).wait()
    output_reshaped = output.reshape(image.shape)

    return output_reshaped


if __name__ == "__main__":
    # Load and Convert image from BGR to GRAY color space.
    img_arr = np.asarray(Image.open('example.jpg')).astype(np.uint8)

    # Define the color stops
    stops = [
        [131, 58, 180],  # Stop 1
        [253, 29, 29],   # Stop 2
        [252, 176, 69]   # Stop 3
    ]

    # Define the number of intermediate colors
    num_intermediate_colors = 127

    palette = generate_gradient_palette(stops, num_intermediate_colors)

    # Map the color palette to the image.
    output_rgb = map_color_palette(img_arr, palette)

    # Convert the output image to PIL Image in BGR format.
    cv2.imwrite("palettized_image.jpg", output_rgb, [
                cv2.IMWRITE_JPEG_QUALITY, 94, cv2.IMWRITE_JPEG_CHROMA_QUALITY, 92])
