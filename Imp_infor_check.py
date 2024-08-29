import csv
from http import HTTPStatus
import dashscope
import time

dashscope.api_key = " "
instruction = " "

def get_completion(prompt_text, retry_count=3, backoff_time=1):
    for attempt in range(retry_count):
        resp = dashscope.Generation.call(
            model='qwen2-7b-instruct',
            prompt=prompt_text
        )
        if resp.status_code == HTTPStatus.OK:
            print(prompt_text)
            print(resp.output)  # The output text
            print(resp.usage)
            return resp.output['text']
        elif resp.status_code == 429:  # Assuming 429 is the rate limit error code
            print(f"Rate limit exceeded, retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Increase backoff time exponentially
        else:
            print(f"Error {resp.code}: {resp.message}")
            return f"Error {resp.code}: {resp.message}"
    return "Max retries exceeded"

def process_document(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as infile, \
            open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for line in reader:
            if line:  # Skip empty lines
                input_text = line[0]  # Assuming input text is in the first column
                prompt = f"{instruction} {input_text}"
                output_text = get_completion(prompt)
                writer.writerow([instruction, input_text, output_text])

if __name__ == '__main__':
    input_file_path = 'data/R_text.csv'
    output_file_path = 'data/RH_text.csv'
    process_document(input_file_path, output_file_path)