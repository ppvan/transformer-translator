import re


def fix_contents(text: str):
    text = re.sub("won &apos;t", "will not", text)
    text = re.sub("can &apos;t", "can not", text)
    text = re.sub("&apos;d", "would", text)
    text = re.sub("&apos;s", "is", text)
    text = re.sub("&apos;re", "are", text)
    text = re.sub("&apos;m", "am", text)
    text = re.sub("&apos;ve", "have", text)
    text = re.sub("&apos;ll", "will", text)
    text = re.sub("&apos;t", "not", text)

    return text
