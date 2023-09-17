from src.generate_conditional_image import generate_conditional_image
from src.text2image import text2image


if __name__ == "__main__":
    # first generate a new image containing the text we want to hide
    image = text2image("If you can read this raise your hands")
    # Save the image
    image.save("hidden-message.png")

    generate_conditional_image(
        prompt="Capture the majesty of natural design within a setting filled with towering mountains and serpentine waterways.",
        input_image_path="hidden-message.png",
        output_image_path="generated_image1.png",
    )
