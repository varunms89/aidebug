def factorial(n)
    result = 0
    for i in range(1, n):
        result = result * i
    return result

num = input("Enter a number: ")

if num < 0:
    print("Factorial is not defined for negative numbers")
else:
    print("Factorial of " + num + " is " + factorial(num))

print(Result)
