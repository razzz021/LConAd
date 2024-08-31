from openai import OpenAI
import os 
import time
import json

client = OpenAI()
shrds = os.listdir("data")


# succuess batch 
GLOBAL_DICT={}

def check_status(batch_input_file_id, max_retries=10, delay=30):
    attempt = 0
    while attempt < max_retries:
        try:
            # 尝试检索批次状态
            status = client.batches.retrieve(batch_input_file_id).status
            return status
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to check status for {batch_input_file_id}: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to check status after {max_retries} attempts.")
                return None  # 或者根据需要处理失败情况

def upload_file(shrd, shrd_id):
    global FAIL_SHRDS
    
    print(f"upload file:{shrd_id}")
    
    batch_input_file = client.files.create(
    file=open(f"data/{shrd}", "rb"),
    purpose="batch"
    )
    
    batch_input_file_id = batch_input_file.id
    
    time.sleep(70)
    
    status = check_status(batch_input_file_id)
    print(f'status file {shrd_id}: {status}')
    
    if status == "in_progress":
        return True
    else:
        return False


def create_batches(batch_input_file_id, shrd):
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": f"lecardv1 summary {shrd}"
        }
    )
    

def write_dict():
    with open("metadata/global_dict.json",'w',encoding="utf8") as f:
        json.dump(GLOBAL_DICT, f)
        


def main():
    shrds = os.listdir("data")
    total_nums = len(shrds)
    retry_delay =60
    max_steps = 10
    for shrd_id, shrd in enumerate(shrds):
        retry_steps = 0
        while retry_steps < max_steps:
            if upload_file(shrd, shrd_id):
                print(f"File {shrd} uploaded successfully.")
                break
            else:
                retry_steps += 1
                print(f"Retry {retry_steps}/{max_steps} for file {shrd}.")
                time.sleep(retry_delay) 
            

        
    
    