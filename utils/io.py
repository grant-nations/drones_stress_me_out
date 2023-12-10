import os
import re


def generate_unique_filename(filename):
    if not os.path.exists(filename):
        return filename

    name, ext = os.path.splitext(filename)
    counter = 1

    while True:
        # strip counter from filename
        stripped_name = re.sub(r'-\d+$', '', name)

        new_filename = f"{stripped_name}-{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1
