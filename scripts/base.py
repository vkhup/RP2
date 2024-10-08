import os
import json
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key="API-KEY")

def query_openai(prompt):
    """
    Queries OpenAI API with a prompt and returns the response from GPT-4.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert scientist within enzyme engineering."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
        max_tokens=1000
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
            full_prompt = prompt_text + "\n\n" + "\n".join(guiding_questions)

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
    prompts_file = f'templates/{prompt_type}_prompts.json'
    output_dir = f'responses/response_{prompt_type}/gpt_{prompt_type}'

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
    output_file = os.path.join(output_dir, f'gpt_responses_{formatted_template_type}_batch_{batch_number}.json')

    # Save the output data with GPT responses back to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"All responses for batch {batch_number} saved to {output_file}")

if __name__ == "__main__":
    main()
