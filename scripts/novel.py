import os
import json
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key="sk-proj-geAAy09XfSVEwsyc8IT3T3BlbkFJKfoxtrRvwxxNCG2vikY7")

def query_openai(prompt):
    """
    Queries OpenAI API with a prompt and returns the response from GPT-4.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a synthetic biology expert with a focus on the DBTL (Design-Build-Test-Learn) pipeline. "
                    "Your task is to propose innovative and scientifically valid solutions, drawing from cutting-edge research and established principles in enzyme engineering and synthetic biology. "
                    "Aim to push the boundaries of current methods while ensuring all suggestions are feasible and rooted in science. "
                    "Provide highly detailed and specific suggestions, including naming concrete factors such as specific genes, proteins, pathways, or tools, and provide detailed explanations for why these were chosen and how they should be implemented."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
        max_tokens=1000,  # Allow for more detailed and comprehensive responses
        temperature=0.5   # Moderate temperature for balanced creativity and precision
    )
    return chat_completion.choices[0].message.content.strip()

def generate_responses(data, batch_number, rounds):
    output_data = {
        "template_type": data["template_type"],
        "gpt_responses": []
    }

    for round_num in range(1, rounds + 1):
        # Iterate through all prompts and generate responses
        for prompt_entry in data['prompts']:
            prompt_id = prompt_entry['id']
            prompt_text = prompt_entry['prompt']
            guiding_questions = prompt_entry.get('guiding_questions', [])

            # Create a comprehensive prompt that includes both the main prompt and guiding questions
            full_prompt = (
                f"{prompt_text}\n\n"
                "Using the guiding questions below, provide a comprehensive, cohesive answer that integrates all relevant points. "
                "Focus on proposing novel, scientifically valid solutions that can be applied within the DBTL pipeline of synthetic biology. "
                "Ensure the response addresses each aspect in detail, exploring innovative approaches and considering potential challenges. "
                "Identify specific factors, such as named genes, proteins, pathways, or other entities, and explain the rationale behind choosing them, including expected impacts and challenges:\n\n"
                + "\n".join(guiding_questions)
            )

            # Generate GPT response for each prompt
            gpt_response = query_openai(full_prompt)

            # Append the result to the output list
            output_data['gpt_responses'].append({
                "id": prompt_id,
                "prompt": prompt_text,
                "guiding_questions": guiding_questions,
                "batch_number": batch_number,
                "round_number": round_num,
                "gpt_response": gpt_response
            })

            # Print message after each response is generated
            print(f"Finished generating response for prompt ID: {prompt_id}, Round: {round_num}")

    return output_data

def format_template_type(template_type):
    """
    Formats the template type to ensure it fits the desired filename convention.
    """
    # Replace spaces with underscores and convert to lowercase
    formatted_type = template_type.replace(" ", "_").lower()
    # Remove any unwanted terms (like 'prompts')
    formatted_type = formatted_type.replace("_prompts", "")
    return formatted_type

def main():
    # Paths to input prompts and output JSON file
    prompts_file = '/Users/venus/PycharmProjects/enzymes/templates/step_by_step_prompts.json'
    output_dir = '/Users/venus/PycharmProjects/enzymes/responses/response_step/gpt_step'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the prompts from the JSON file
    with open(prompts_file, 'r') as file:
        data = json.load(file)

    # Extract and format template type
    template_type = data.get("template_type", "default_template")
    formatted_template_type = format_template_type(template_type)

    # Parameters for batch number and rounds
    batch_number = 1
    rounds = 10  # Define how many rounds of responses you want to generate

    # Generate responses
    output_data = generate_responses(data, batch_number, rounds)

    # Create an output file for this batch using the formatted template type
    output_file = os.path.join(output_dir, f'novel_gpt_responses_{formatted_template_type}_batch_{batch_number}.json')

    # Save the output data with GPT responses back to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"All responses for batch {batch_number} saved to {output_file}")

if __name__ == "__main__":
    main()