# -*- coding: utf-8 -*-

import datetime
import os
import subprocess


def genRequirementsTxt(dir: str) -> None:
    """基于pipreqs库生成针对dir目录的requirements.txt文件，并更新版本号"""
    try:
        # 使用pipreqs生成requirements.txt文件
        result = subprocess.run(
            ["pipreqs", dir, "--force", "--encoding=utf-8"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        requirements_file = os.path.join(dir, "requirements.txt")

        if result.returncode == 0:
            print("%s: Requirements.txt file generated successfully." % cur_time)
            print("Output:\n", result.stdout)
            # 读取生成的requirements.txt文件
            with open(requirements_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # 更新版本号
            updated_lines = []
            for line in lines:
                package = line.split("==")[0]
                version_result = subprocess.run(
                    ["pip", "show", package],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if version_result.returncode == 0:
                    for version_line in version_result.stdout.splitlines():
                        if version_line.startswith("Version:"):
                            version = version_line.split(" ")[1]
                            updated_lines.append(f"{package}=={version}\n")
                            break
                else:
                    updated_lines.append(line)
            # 写回更新后的requirements.txt文件
            with open(requirements_file, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)
            print("%s: Requirements.txt file updated with correct versions." % cur_time)
        else:
            print("%s: Failed to generate requirements.txt file." % cur_time)
            print("Error:\n", result.stderr)
    except Exception as e:
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        print("%s: Failed to generate requirements.txt file, error: %s" % (cur_time, e))


if __name__ == "__main__":
    genRequirementsTxt("./")
