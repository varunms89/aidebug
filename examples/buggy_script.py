"""Sample buggy script for testing `aidebug run examples/buggy_script.py`."""


def average(numbers):
    total = 0
    for n in numbers:
        total += int(n)
    return total / len(numbers)


def main():
    scores = [10, 20, "30", 40]
    print("Average score:", average(scores))


if __name__ == "__main__":
    main()
