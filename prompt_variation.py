import os, subprocess

def prompt_variation(prompt, n_variations):
    
    # option 1
    # complete_prompt = f"""Generate {str(int(n_variations))} reformulations of the prompt delimited by 3 semicolons.
    # Keep the same style as the input, just make slight variations. Remove the semicolons from the output.
    # Format the output as a list of sentences that start with a -.
    
    # ;;;{prompt};;;"""

    # option 2
    complete_prompt = f"""Generate {str(int(n_variations))} reformulations of the prompt delimited by 3 semicolons.
    Each reformulation should be the same as the prompt, except that one word should be changed, removed or added. The meaning should remain the same. 
    Remove the semicolons from the output. Format the output as a list of sentences that start with a -. Don't explain the changes. 
    Return the reformulations only. Don't return the prompt. Don't say anything but the reformulations.
    
    ;;;{prompt};;;"""


    # get the output from the model using CLI
    response = subprocess.check_output(["llm", "-m", "llama2", complete_prompt])
    response = response.decode("utf-8")
    response = response.split("\n")
    
    # get the reformulations
    reformulations = []
    for line in response:
        if line.startswith("- "):
            reformulations.append(line[2:])
    assert len(reformulations) == n_variations, f"Expected {n_variations} reformulations, but got {len(reformulations)}."
    
    return reformulations
    
if __name__ == "__main__":
    
    prompt = """A blue river with red stones."""
    
    print(prompt_variation(prompt, 5))
