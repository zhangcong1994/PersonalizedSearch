import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "t2ranking", "collection.tsv")

lengths = []
html_count = 0
with open(data_path, "r", encoding="utf-8") as f:
    header = f.readline()
    print(f"Header: {repr(header.strip())}")

    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        if i < 5:
            text = parts[1] if len(parts) > 1 else ""
            print(f"\nLine {i}: pid={parts[0]}, text_len={len(text)}")
            print(f"  Text[:200]: {text[:200]}")

        if len(parts) > 1:
            t = parts[1]
            lengths.append(len(t))
            if "<br" in t or "<img" in t or "<div" in t or "<p>" in t:
                html_count += 1

print(f"\n{'='*50}")
print(f"Total passages: {len(lengths):,}")
print(f"Passages with HTML tags: {html_count:,} ({html_count/len(lengths)*100:.1f}%)")

if lengths:
    lengths.sort()
    print(f"  Min length:    {min(lengths)}")
    print(f"  Max length:    {max(lengths)}")
    print(f"  Median length: {lengths[len(lengths)//2]}")
    print(f"  Avg length:    {sum(lengths)/len(lengths):.0f}")
    print(f"  P25: {lengths[len(lengths)//4]}, P75: {lengths[len(lengths)*3//4]}, P90: {lengths[len(lengths)*9//10]}")
    gt500 = sum(1 for l in lengths if l > 500)
    gt1000 = sum(1 for l in lengths if l > 1000)
    gt2000 = sum(1 for l in lengths if l > 2000)
    print(f"  >500 chars:  {gt500:,} ({gt500/len(lengths)*100:.1f}%)")
    print(f"  >1000 chars: {gt1000:,} ({gt1000/len(lengths)*100:.1f}%)")
    print(f"  >2000 chars: {gt2000:,} ({gt2000/len(lengths)*100:.1f}%)")
