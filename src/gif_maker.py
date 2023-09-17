from PIL import Image, ImageSequence
import os


def aggregate_images_to_gif(list_img, output_path, duration=200, loop=1, sort=True):
    """Aggregate a list of images into a gif
    
    Args:
        list_img (list): list of image paths
        output_path (str): path to save the gif
        duration (int, optional): duration of each frame in milliseconds. Defaults to 200.
        loop (int, optional): number of loops. Defaults to 1.
        sort (bool, optional): sort the images by basename. Defaults to True.
    """
    
    # sort by name 
    if sort:
        image_files.sort(key=lambda x: int(os.path.basename(x).replace("f", "").replace(".png", "")))

    # Create a list to hold the image objects
    images = []
    
    # Open and append each image to the list
    for filename in list_img:
        image = Image.open(filename)
        images.append(image)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,  # Specify the duration (in milliseconds) for each frame
        loop=loop,  # 0 means infinite loop; you can change this number to loop a specific number of times
    )

if __name__ == "__main__":

    #list images
    img = os.listdir("/Users/ymdt/src/subliminAI/autoimagemorph")
    image_files = ["/Users/ymdt/src/subliminAI/autoimagemorph/"+i for i in img if i.endswith(".png")]
    
    aggregate_images_to_gif(image_files, "output.gif", duration=10, loop=1)