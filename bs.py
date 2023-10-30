import sys

# Function to add two numbers
def add(x, y):
    return x + y

# Function to subtract two numbers
def subtract(x, y):
    return x - y

# Function to multiply two numbers
def multiply(x, y):
    return x * y

# Function to divide two numbers
def divide(x, y):
    if y == 0:
        return "Division by zero is not allowed."
    return x / y

if len(sys.argv) != 4:
    print("Usage: python bs.py <operation> <number1> <number2>")
    sys.exit(1)

operation = sys.argv[1]
num1 = float(sys.argv[2])
num2 = float(sys.argv[3])

if operation == "add":
    result = add(num1, num2)
elif operation == "subtract":
    result = subtract(num1, num2)
elif operation == "multiply":
    result = multiply(num1, num2)
elif operation == "divide":
    result = divide(num1, num2)
else:
    result = "Invalid operation. Please use 'add', 'subtract', 'multiply', or 'divide'."

print("Result:", result)
