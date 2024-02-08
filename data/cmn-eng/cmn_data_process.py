import os

new_lines = []
with open('./cmn.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line_arr = line.strip().split('\t')
        new_lines.append('\t'.join(line_arr[:2]))
with open('./cmn_data_process.txt', 'w', encoding='utf-8') as f:
    for line in new_lines:
        f.write(line + '\n')
