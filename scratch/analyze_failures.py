import re

log_path = "/Users/yujingchen/code/lilith-agent/.lilith/failed_tasks_trace.log"

task_summaries = {}
current_task = None

task_start_pattern = re.compile(r"\[runner\] task=([0-9a-f\-]+) \(\d+/\d+\) starting")
summary_pattern = re.compile(r"summary='(.*)'$")
answer_pattern = re.compile(r"\[runner\] task=([0-9a-f\-]+) \(\d+/\d+\) answer='(.*)'")
content_pattern = re.compile(r"\[model\] finished content='(.*)'")
prompt_q_pattern = re.compile(r"q='([^']*)'")

with open(log_path, "r") as f:
    for line in f:
        start_match = task_start_pattern.search(line)
        if start_match:
            current_task = start_match.group(1)
            if current_task not in task_summaries:
                task_summaries[current_task] = {"summary": "", "content": "", "answer": "", "q": ""}
            
            # Try to grab the truncated question
            q_match = prompt_q_pattern.search(line)
            if q_match:
                task_summaries[current_task]["q"] = q_match.group(1)
            continue
            
        if current_task:
            summary_match = summary_pattern.search(line)
            if summary_match:
                task_summaries[current_task]["summary"] = summary_match.group(1)
                
            content_match = content_pattern.search(line)
            if content_match:
                # keep updating content because there might be multiple calls, we want the last one before answer
                task_summaries[current_task]["content"] = content_match.group(1)
                
            answer_match = answer_pattern.search(line)
            if answer_match:
                t_id = answer_match.group(1)
                ans = answer_match.group(2)
                if t_id in task_summaries:
                    task_summaries[t_id]["answer"] = ans

with open("/Users/yujingchen/code/lilith-agent/scratch/failed_tasks_analysis.md", "w") as out:
    for t_id, data in task_summaries.items():
        out.write(f"### Task ID: {t_id}\n")
        out.write(f"**Question (truncated):** {data['q']}\n")
        out.write(f"**Agent Final Content:** {data['content']}\n")
        out.write(f"**Extracted Answer:** {data['answer']}\n")
        out.write(f"**Episode Summary:** {data['summary']}\n\n")

print(f"Extracted analysis for {len(task_summaries)} tasks.")
