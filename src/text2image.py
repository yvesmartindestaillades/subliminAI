from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "arial": "fonts/arial.ttf",
    "arialbd": "fonts/arialbd.ttf",
    "arialbi": "fonts/arialbi.ttf",
}


def text2image(
    text,
    font_size=100,
    font_color=(0, 0, 0),
    background_color=(255, 255, 255),
    width=512,
    height=512,
    text_height=None,
    text_width=None,
    font="arialbd",
):
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

    # Initialize variables to hold text size and position
    text_width, text_height = draw.textsize(text, font=font)

    # Prepare text for drawing
    words = text.split()
    lines = []
    current_line = []
    current_line_width = 0

    for word in words:
        word_width, word_height = draw.textsize(word, font=font)
        if current_line_width + word_width <= width:
            current_line.append(word)
            current_line_width += word_width + font_size // 4  # space width
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_line_width = word_width + font_size // 4  # space width

    lines.append(" ".join(current_line))

    # Calculate total text height
    total_text_height = len(lines) * word_height

    # Draw text line by line
    y_text = (height - total_text_height) // 2
    for line in lines:
        line_width, _ = draw.textsize(line, font=font)
        # Calculate X position
        x_text = (width - line_width) // 2
        draw.text((x_text, y_text), line, font=font, fill="black")
        y_text += word_height

    return image


if __name__ == "__main__":
    # Convert text to image
    image = text2image("If you can read this raise your hands")

    # Save the image
    image.save("hello.png")
    image.show()
