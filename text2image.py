from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "arial": "fonts/arial.ttf",
    "arialbd": "fonts/arialbd.ttf",
    "arialbi": "fonts/arialbi.ttf",
}

def text2image(text, font_size=30, font_color=(0, 0, 0), background_color=(255, 255, 255), width=400, height=200, text_height=None, text_width=None, font="arial"):
    """Convert text to an image.

    Args:
        text (str): Text to be converted.
        font_size (int): Font size.
        font_color (tuple): Font color in RGB format.
        background_color (tuple): Background color in RGB format.
        width (int): Width of the image.
        height (int): Height of the image.
        text_height (int): Height of the text. If None, it will be calculated automatically.
        text_width (int): Width of the text. If None, it will be calculated automatically.
        font (str): Font name. It can be one of the following: "arial", "arialbd", "arialbi".

    Returns:
        Image: An image containing the text.
    """
    # Create a blank image with the desired dimensions and background color
    image = Image.new("RGB", (width, height), background_color)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define font settings
    font = ImageFont.truetype(FONTS[font], font_size)

    # Get text size
    text_width, text_height = draw.textsize(text, font)

    # Calculate text position (centered)
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw the text on the image
    draw.text((x, y), text, font=font, fill=font_color)

    return image

if __name__ == "__main__":
    # Convert text to image
    image = text2image("Hello, world!")

    # Save the image
    image.save("hello.png")
    image.show()