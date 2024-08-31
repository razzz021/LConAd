import argparse
import jsonlines 

def read_jsonlines(file):
    with jsonlines.open(file, 'r') as reader:
        return list(reader)

def jsonl_to_dict(ls):
    res = {}
    for l in ls:
        text = l['text']
        text = text.strip("：").strip("，").strip()
        res[l['text_id']] = text
    return res  

def single(input, output):

    corpus = read_jsonlines(input)
    corpus = jsonl_to_dict(corpus)
    corpus = list(corpus.values())
    
    with open(output, 'w', encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")
     

def combined(input1, input2, output):

    corpus1 = read_jsonlines(input1)
    corpus2 = read_jsonlines(input2)
    
    corpus1 = jsonl_to_dict(corpus1)
    corpus2 = jsonl_to_dict(corpus2)
    
    res = []
    for key in corpus1:
        if key in corpus2:
            res.append(corpus1[key])
            res.append(corpus2[key])
    
    with open(output, 'w', encoding="utf-8") as f:
        for line in res:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process some JSONL files.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the 'single' command
    parser_single = subparsers.add_parser('single', help='Process a single JSONL file')
    parser_single.add_argument('--input', type=str, required=True, help='Path to the input JSONL file')
    parser_single.add_argument('--output', type=str, required=True, help='Path to the output file')

    # Subparser for the 'combined' command
    parser_combined = subparsers.add_parser('combined', help='Combine two JSONL files')
    parser_combined.add_argument('--input1', type=str, required=True, help='Path to the first input JSONL file')
    parser_combined.add_argument('--input2', type=str, required=True, help='Path to the second input JSONL file')
    parser_combined.add_argument('--output', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()

    if args.command == 'single':
        single(args.input, args.output)
        
        
    elif args.command == 'combined':
        combined(args.input1, args.input2, args.output)
        
        
    else:
        parser.print_help()
        
    # useage python script.py single --input path/to/input.jsonl --output path/to/output.txt
    
if __name__=="__main__":
    main()
