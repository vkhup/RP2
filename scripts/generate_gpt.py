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
            {"role": "system", "content": "You are a helpful assistant for enzyme engineering."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4",
    )

    return chat_completion.choices[0].message.content.strip()


def main():
    # Paths to input prompts and output JSON file
    prompts_file = '/Users/venus/PycharmProjects/enzymes/templates/cause_effect/cause_effect_prompts.json'
    output_file = '/Users/venus/PycharmProjects/enzymes/responses/response_cause_effect/gpt_cause_effect/gpt_responses_cause_effect.json'

    # Load the prompts from the JSON file
    with open(prompts_file, 'r') as file:
        data = json.load(file)

    # Prepare a list to hold the output
    output_data = {
        "template_type": data["template_type"],
        "gpt_responses": []
    }

    # Iterate through all prompts and generate responses
    for prompt_entry in data['prompts']:
        prompt_id = prompt_entry['id']
        prompt_text = prompt_entry['prompt']

        # Generate GPT response for each prompt
        gpt_response = query_openai(prompt_text)

        # Append the result to the output list
        output_data['gpt_responses'].append({
            "id": prompt_id,
            "prompt": prompt_text,
            "gpt_response": gpt_response
        })

        # Print message after each response is generated
        print(f"Finished generating response for prompt ID: {prompt_id}")

    # Save the output data with GPT responses back to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"All responses saved to {output_file}")


if __name__ == "__main__":
    main()