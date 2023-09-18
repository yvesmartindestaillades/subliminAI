from src.generate_conditional_image import generate_conditional_image
from src.text2image import text2image
from src.prompt_variation import prompt_variation
from morphing import generate_morphing_between_images


if __name__ == "__main__":
    # first generate a new image containing the text we want to hide
    image = text2image("If you can read this raise your hands")
    # Save the image
    image.save("hidden-message.png")

    user_prompt = "A beautiful italian landscape. Houses, trees, a river and a bridge."

    prompt_variations = prompt_variation(user_prompt, n_variations=5)

    images_for_morphing = []
    for idx, prompt in enumerate(prompt_variations):
        # generate image
        out_name = f"generated_image_{idx}.png"
        generate_conditional_image(
            prompt=prompt,
            input_image_path="hidden-message.png",
            output_image_path=out_name,
        )
        images_for_morphing.append(out_name)

    # morphing between images
    generate_morphing_between_images(images_for_morphing)
