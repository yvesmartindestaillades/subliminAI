import argparse
from src.generate_animation import GenerateAnimation

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

    GenerateAnimation().run(
        args.text,
        args.prompt,
        args.control_strength,
        args.number_of_frames,
        args.use_morphing,
    )
