from generate_conditional_image import generate_conditional_image
from text2image import text2image
from prompt_variation import prompt_variation
from morphing import generate_morphing_between_images
from gif_maker import aggregate_images_to_gif

import shutil, os

class GenerateAnimation:
    def __init__(self):
        self.state = None
        self.progress_bar_val = 1
        self.progress_bar_val_max = 1
        
    def run(
        self,
        text: str,
        prompt: str,
        control_strength: float,
        number_of_frames: int,
        use_morphing: bool,
        output_path: str='out/output.gif'
    ) -> None:
        print(
            f"""
    text: {text}
    prompt: {prompt}
    control_strength: {control_strength}
    number_of_frames: {number_of_frames}"""
        )

        self.progress_bar_val_max = number_of_frames

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
            self.state = f"Generating image {idx+1} of {len(prompt_variations)}..."
            self.progress_bar_val = idx+1
            images_for_morphing.append(out_name)

        os.makedirs("out", exist_ok=True)
        # morphing between images
        if len(images_for_morphing) == 1:
            shutil.copy(images_for_morphing[0], output_path)
        else:
            if use_morphing:
                generate_morphing_between_images(images_for_morphing)

            # turn images into gif
            # list images
            img = os.listdir("img")
            image_files = ["img/" + i for i in img if i.endswith(".png")]

            self.state = "Generating gif..."
            # save gif
            aggregate_images_to_gif(image_files, output_path, duration=10, loop=0)

        self.state = "Done"