

def solve_quadratic_equation(a,b,c):
    """Solve a quadratic equation using the quadratic formula."""
    import math
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solutions")
    else:
        return ((-b + math.sqrt(discriminant))/(2*a),
                (-b - math.sqrt(discriminant))/(2*a))


class fully_connected_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(fully_connected_network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x