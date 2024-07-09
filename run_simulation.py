import subprocess
import sys

n = int(sys.argv[1])

win_counters = [0, 0, 0, 0, 0]
for _ in range(n):
    result = subprocess.run(
        ["python3", "match_simulator.py", "--submissions", "2:submission_4.py","3:submission_8.py", "--engine"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output_lines = result.stdout.splitlines()

    if len(output_lines) >= 8:
        # Get the 8th last line
        eighth_last_line = output_lines[-8]

        if "SUCCESS" in eighth_last_line:
            # Extract the ranking list
            ranking_str = eighth_last_line.split("ranking=")[1].strip("[]}")
            ranking = list(map(int, ranking_str.split(',')))

            win_counters[ranking[0]] += 1

        # Write the 8th last line to the output file
        with open("testing.out", "a") as file:
            file.write(eighth_last_line + "\n")
    else:
        with open("testing.out", "a") as file:
            file.write("Not enough lines in output\n")

for bot in range(5):
    print(f"Wins of bot {bot}: {win_counters[bot]}")
