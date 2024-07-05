import subprocess
import sys

n = int(sys.argv[1])

output_file = "testing.out"

for _ in range(n):
    result = subprocess.run(
        ["python3", "match_simulator.py", "--submissions", "4:submission_2.py", "1:submission_3.py", "--engine"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output_lines = result.stdout.splitlines()

    if len(output_lines) >= 8:
        # Get the 8th last line
        eighth_last_line = output_lines[-8]
    else:
        eighth_last_line = "Not enough lines in output"

    # Write the 8th last line to the output file
    with open(output_file, "a") as file:
        file.write(eighth_last_line + "\n")
