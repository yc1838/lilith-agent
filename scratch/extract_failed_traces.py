import re

failed_tasks_str = "14569e28-c88c-43e4-8c32-097d35b9a67d,7dd30055-0198-452e-8c25-f73dbe27dcb8,624cbf11-6a41-4692-af9c-36b3e5ca3130,dd3c7503-f62a-4bd0-9f67-1b63b94194cc,f0f46385-fc03-4599-b5d3-f56496c3e69f,56137764-b4e0-45b8-9c52-1866420c3df5,8b3379c0-0981-4f5b-8407-6444610cb212,a7feb290-76bb-4cb7-8800-7edaf7954f2f,b4cc024b-3f5e-480e-b96a-6656493255b5,b9763138-c053-4832-9f55-86200cb1f99c,16d825ff-1623-4176-a5b5-42e0f5c2b0ac,bfcd99e1-0690-4b53-a85c-0174a8629083,08cae58d-4084-4616-b6dd-dd6534e4825b,2dfc4c37-fec1-4518-84a7-10095d30ad75,ecbc4f94-95a3-4cc7-b255-6741a458a625,48eb8242-1099-4c26-95d4-ef22b002457a,08f3a05f-5947-4089-a4c4-d4bcfaa6b7a0,54612da3-fd56-4941-80f4-5eb82330de25,ded28325-3447-4c56-860f-e497d6fb3577,6359a0b1-8f7b-499b-9336-840f9ab90688,0a3cd321-3e76-4622-911b-0fda2e5d6b1a,f2feb6a4-363c-4c09-a804-0db564eafd68,0b260a57-3f3a-4405-9f29-6d7a1012dbfb,cca70ce6-1952-45d2-acd4-80c903b0bc49,023e9d44-96ae-4eed-b912-244ee8c3b994,0e9e85b8-52b9-4de4-b402-5f635ab9631f,20194330-9976-4043-8632-f8485c6c71b2,65638e28-7f37-4fa7-b7b9-8c19bb609879,708b99c5-e4a7-49cb-a5cf-933c8d46470d,d5141ca5-e7a0-469f-bf3e-e773507c86e2,b2c257e0-3ad7-4f05-b8e3-d9da973be36e,db4fd70a-2d37-40ea-873f-9433dc5e301f,7a4a336d-dcfa-45a0-b014-824c7619e8de"
failed_tasks = set(failed_tasks_str.split(","))

log_path = "/Users/yujingchen/code/lilith-agent/.lilith/session-20260510-182256.log"
output_path = "/Users/yujingchen/code/lilith-agent/.lilith/failed_tasks_trace.log"

current_task = None
task_start_pattern = re.compile(r"\[runner\] task=([0-9a-f\-]+) \(\d+/\d+\) starting")

with open(log_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        match = task_start_pattern.search(line)
        if match:
            current_task = match.group(1)
        
        if current_task in failed_tasks:
            f_out.write(line)

print("Done extracting.")
