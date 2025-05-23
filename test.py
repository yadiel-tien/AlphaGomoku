import re

input_path = r"C:\Users\cmc\.config\clash\profiles\1744613524025.yml"
output_path = r"C:\Users\cmc\.config\clash\profiles\filtered_clash_config.yml"


def filter_group_proxy_lines(text):
    lines = text.splitlines()
    output_lines = []

    in_proxy_group = False

    for line in lines:
        if line == 'proxy-groups:':
            in_proxy_group = True
            output_lines.append(line)
            continue

        if line == 'rules:':
            in_proxy_group = False

        if not in_proxy_group:
            output_lines.append(line)
            continue

        parts = line.split(',')
        filtered_part = []
        for part in parts:
            if '香港' not in part:
                filtered_part.append(part)
        output_lines.append(','.join(filtered_part))

    return "\n".join(output_lines)


def main():
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    filtered = filter_group_proxy_lines(raw_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(filtered)

    print(f"[SUCCESS] 已过滤 group 中的香港节点，保存到: {output_path}")


if __name__ == "__main__":
    main()
