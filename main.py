from src.generate_conditional_image import generate_conditional_image
from src.text2image import text2image
from src.prompt_variation import prompt_variation
from src.morphing import generate_morphing_between_images
from src.gif_maker import aggregate_images_to_gif
from src import prompts

import shutil, os

max_img = 10

if __name__ == "__main__":
    # first generate a new image containing the text we want to hide
    image = text2image("Raise your hands")
    # Save the image
    image.save("hidden-message.png")

    user_prompt = "A forest full of mystery and magic. Elves and fairies live here. A blue river with red stones."

    #prompt_variations = prompt_variation(user_prompt, n_variations=10)
    prompt_variations = prompts.forest
    
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
            control_net_strength=1.3
        )
        images_for_morphing.append(out_name)
        if idx > max_img:
            break

    # morphing between images
    #generate_morphing_between_images(images_for_morphing)

    # turn images into gif
    # list images
    img = os.listdir("img")
    image_files = ["img/" + i for i in img if i.endswith(".png")]
    
    # save gif
    os.makedirs("out", exist_ok=True)
    aggregate_images_to_gif(image_files, "out/output.gif", duration=10, loop=0)
