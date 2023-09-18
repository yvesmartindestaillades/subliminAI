import replicate
import requests
from typing import Optional


def download_image(url: str, local_filename: str) -> None:
    """
    Download an image from a given URL and save it to a local file.

    Parameters:
        url (str): URL of the image to download.
        local_filename (str): Local file path to save the downloaded image.

    Returns:
        None
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download image: {response.status_code}")


def call_replicate_api(
    prompt: str, image_path: str, control_net_strength: float = 0.8
) -> Optional[str]:
    """
    Call the Replicate API to generate a conditional image based on a given prompt and settings.

    Parameters:
        prompt (str): Textual prompt for the generated image.
        image_path (str): Path to the input image.
        control_net_strength (float): Strength of the control network. Default is 0.8.

    Returns:
        Optional[str]: URL of the generated image, or None if unsuccessful.
    """
    with open(image_path, "rb") as file:
        response = replicate.run(
            "andreasjansson/qrcode:75d51a73fce3c00de31ed9ab4358c73e8fc0f627dc8ce975818e653317cb919b",
            input={
                "prompt": prompt,
                "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
                "qrcode_background": "white",
                "image": file,
                "seed": 42,
                "height": 512,
                "width": 512,
                "num_inference_steps": 40,
                "controlnet_conditioning_scale": control_net_strength,
                "qr_code_content": "https://www.youtube.com/watch?v=lW1RAYYs8RI",
            },
        )

        return response[0] if response else None


def generate_conditional_image(
    prompt: str,
    input_image_path: str,
    output_image_path: str,
    control_net_strength: float = 0.8,
) -> None:
    """
    Generate a conditional image using the Replicate API and save it to a local file.

    Parameters:
        prompt (str): Textual prompt for the generated image.
        input_image_path (str): Path to the input image.
        output_image_path (str): Local file path to save the generated image.
        control_net_strength (float): Strength of the control network. Default is 0.8.

    Returns:
        None
    """
    output_url = call_replicate_api(prompt, input_image_path, control_net_strength)
    if output_url:
        download_image(output_url, output_image_path)


if __name__ == "__main__":
    generate_conditional_image(
        prompt="Capture the majesty of natural design within a setting filled with towering mountains and serpentine waterways.",
        input_image_path="hidden-message-big.png",
        output_image_path="generated_image1.png",
    )
