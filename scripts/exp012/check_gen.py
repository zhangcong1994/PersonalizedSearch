"""检查 generation 文件格式。"""
import json, sys
sys.path.insert(0, ".")

from src.utils.config import DATA_ROOT

gen_file = DATA_ROOT / "results/exp011/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl"

with open(gen_file, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total records: {len(lines)}")
r0 = lines[0]
print(f"\nKeys: {sorted(r0.keys())}")
print(f"query_id: {r0['query_id']}")
print(f"original_query_id: {r0.get('original_query_id')}")
print(f"sample_id: {r0.get('sample_id')}")
print(f"model_id: {r0.get('model_id')}")
print(f"temperature: {r0.get('temperature')}")
print(f"has system_prompt: {'system_prompt' in r0}")
print(f"has passages: {'passages' in r0}")
print(f"passages count: {len(r0.get('passages', []))}")
print(f"has answer: {'answer' in r0}")
answer = r0.get('answer', '')
print(f"answer length: {len(answer)}")

# passages 第一条
if r0.get('passages'):
    p0 = r0['passages'][0]
    print(f"\npassage[0] keys: {sorted(p0.keys())}")
    print(f"  rank: {p0.get('rank')}, pid: {p0.get('pid')}, text_len: {len(p0.get('text',''))}")

# system_prompt 预览
sp = r0.get('system_prompt', '')
print(f"\nsystem_prompt preview (first 200 chars): {sp[:200]}")
