import os
import argparse
import random
import json

def apply_qa_template(data):
  return f"""Given the following table context
{data["context"]}
Question: {data["question"]}
Answer: {data["answer"]}"""

def transform_data(data):
  return { "text": apply_qa_template(data) }

def split(data, validation_split_ratio, test_split_ratio):
  random.shuffle(data)
  test_split_index = int(test_split_ratio * len(data))
  train_data = data[test_split_index:]
  test_data = data[:test_split_index]

  validation_split_index = int(validation_split_ratio * len(train_data))
  validation_data = train_data[:validation_split_index]
  train_data = train_data[validation_split_index:]

  return train_data, validation_data, test_data

def build_parser():
  argparser = argparse.ArgumentParser(description='Preprocess SQL training data')
  argparser.add_argument('--file', '-f', required=True, help='Path to the training data file')
  argparser.add_argument('--validation-split-ratio', '-vsr', type=float, default=0.1, help='Ratio of validation data')
  argparser.add_argument('--test-split-ratio', '-tsr', type=float, default=0.2, help='Ratio of test data')
  argparser.add_argument('--output-dir', '-o', default='data', help='Output directory for the preprocessed data')
  return argparser

def prepare_output_dir(output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def write_jsonl(file, data):
  with open(file, 'w') as f:
    for line in data:
      f.write(json.dumps(line) + '\n')

def read_json_file(file):
  with open(file, 'r') as f:
    return json.load(f)


OUTPUT_TRAIN_FILE = 'train.jsonl'
OUTPUT_VALIDATION_FILE = 'valid.jsonl'
OUTPUT_TEST_FILE = 'test.jsonl'

def preprocess_data(file, validation_split_ratio, test_split_ratio, output_dir):
  # Read the data from the file
  parsed_json = read_json_file(file)
  print(f"Read {len(parsed_json)} lines of data")

  transformed_data = [transform_data(data) for data in parsed_json]
  train_data, validation_data, test_data = split(transformed_data, validation_split_ratio, test_split_ratio)

  print(f"Split the data into {len(train_data)} lines of training data, {len(validation_data)} lines of validation data, and {len(test_data)} lines of test data")

  prepare_output_dir(output_dir)
  train_file = os.path.join(output_dir, OUTPUT_TRAIN_FILE)
  validation_file = os.path.join(output_dir, OUTPUT_VALIDATION_FILE)
  test_file = os.path.join(output_dir, OUTPUT_TEST_FILE)

  write_jsonl(train_file, train_data)
  write_jsonl(validation_file, validation_data)
  write_jsonl(test_file, test_data)


def main():
  parser = build_parser()
  args = parser.parse_args()
  preprocess_data(args.file, args.validation_split_ratio, args.test_split_ratio, args.output_dir)

if __name__ == '__main__':
  main()
