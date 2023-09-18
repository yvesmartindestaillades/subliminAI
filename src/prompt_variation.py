import os, subprocess, replicate


def prompt_variation(prompt, n_variations):
    if n_variations <= 1:
        return [prompt]
    # option 1
    # complete_prompt = f"""Generate {str(int(n_variations))} reformulations of the prompt delimited by 3 semicolons.
    # Keep the same style as the input, just make slight variations. Remove the semicolons from the output.
    # Format the output as a list of sentences that start with a -.

    # ;;;{prompt};;;"""

    # option 2
    complete_prompt = f"""
    Follow the instructions below:
    
    step1: Generate {str(int(n_variations))} reformulations of the prompt delimited by 3 semicolons. The meaning should remain the same. 
    Remove the semicolons from the output. Format the output as a list of sentences that start with a -. Don't explain the changes. 
    Return the reformulations only. Don't return the prompt. Don't say anything but the reformulations. 
    
    step2: Count the reformulations, and if there are not {str(int(n_variations))} reformulations, add some more.
    
    ;;;{prompt};;;"""

    # get the output from the model using CLI
    response = replicate.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={"prompt": complete_prompt},
    )

    # clean the response
    response = "".join([i for i in response])
    response = response.split("\n")

    # remove the prompt
    reformulations = []
    for item in response:
        if item.startswith("- "):
            reformulations.append(item[2:])

    # get the reformulations
    if not len(reformulations) == n_variations:  # TODO: raise error
        print(
            f"Expected {n_variations} reformulations, but got {len(reformulations)}"
        )  # . response: {response}")

    return reformulations


if __name__ == "__main__":
    prompt = """A blue river with red stones."""

    print(prompt_variation(prompt, 5))
