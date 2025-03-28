import sys
import subprocess


if len(sys.argv) > 1:
    if (sys.argv[1] == '0' or sys.argv[1]=='push'):
        if (sys.argv[2] and sys.argv[2]!=''):
            cmd = [
                "git add .",
                f"git commit -m \"{sys.argv[2]}\"",
                "git push -u origin main"
            ]
        else:
            cmd = [
                "git add .",
                f"git commit -m \"Updating files\"",
                "git push -u origin main"
            ]
        for c in cmd:
            process = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            print(stdout, stderr)


    elif (sys.argv[1] == '1' or sys.argv[1]=='pull'):
        result = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True)
        print(result.stdout, result.stderr)
    else :
        raise Exception("No arguments provided.")

else:
    raise Exception("No arguments provided.")