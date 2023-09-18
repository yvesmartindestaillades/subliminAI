import argparse

from src.generate_conditional_image import generate_conditional_image
from src.text2image import text2image
from src.prompt_variation import prompt_variation
from src.morphing import generate_morphing_between_images
from src.gif_maker import aggregate_images_to_gif
from src import prompts

import shutil, os


def generate_animation(
    text: str,
    prompt: str,
    control_strength: float,
    number_of_frames: int,
    use_morphing: bool,
) -> None:
    print(
        f"""
text: {text}
prompt: {prompt}
control_strength: {control_strength}
number_of_frames: {number_of_frames}"""
    )

    # first generate a new image containing the text we want to hide
    image = text2image(text)
    # Save the image
    image.save("hidden-message.png")

    prompt_variations = prompt_variation(prompt, n_variations=number_of_frames)

    # remove /img folder
    if os.path.exists("img"):
        shutil.rmtree("img")
    os.makedirs("img")

    images_for_morphing = []
    for idx, prompt in enumerate(prompt_variations):
        # generate image
        out_name = f"img/generated_image_{idx}.png"
        generate_conditional_image(
            prompt=prompt,
            input_image_path="hidden-message.png",
            output_image_path=out_name,
            control_net_strength=control_strength,
        )
        images_for_morphing.append(out_name)

    os.makedirs("out", exist_ok=True)
    # morphing between images
    if len(images_for_morphing) == 1:
        shutil.copy(images_for_morphing[0], f"out/output.png")
    else:
        if use_morphing:
            generate_morphing_between_images(images_for_morphing)

        # turn images into gif
        # list images
        img = os.listdir("img")
        image_files = ["img/" + i for i in img if i.endswith(".png")]

        # save gif
        aggregate_images_to_gif(image_files, "out/output.gif", duration=10, loop=0)


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Generate single image or animation.")

    # Add arguments
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="If you can read this raise your hands",
        help="Text to generate the image or animation.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="A forest full of mystery and magic. Elves and fairies live here. A blue river with red stones.",
        help="Prompt for the image or animation.",
    )
    parser.add_argument(
        "-c",
        "--control_strength",
        type=float,
        default=0.8,
        help="Control strength for the image or animation. Between 0.0 and 4.0. Default: 0.8",
    )
    parser.add_argument(
        "-n",
        "--number_of_frames",
        type=int,
        default=1,
        help="Number of frames for animation. Defaults to 1 (single image).",
    )
    parser.add_argument(
        "-m",
        "--use_morphing",
        action="store_true",
        help="Enable morphing between frames.",
    )

    # Parse arguments
    args = parser.parse_args()

    generate_animation(
        args.text,
        args.prompt,
        args.control_strength,
        args.number_of_frames,
        args.use_morphing,
    )
