import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import curve_fit
import sympy

# smaller subset of function if you need
# # Define your symbolic function library 
# torch_functions = {
#     'sin': lambda x, a, b, c, d: c * torch.sin(a * x + b) + d,
#     'exp': lambda x, a, b, c, d: c * torch.exp(a * x + b) + d,
#     'poly2': lambda x, a, b, c, d: a * x**2 + b * x + c + d,
#     # ... Add more functions as needed
# }

# sympy_functions = {
#     'sin': lambda x, a, b, c, d: c * sympy.sin(a * x + b) + d,
#     'exp': lambda x, a, b, c, d: c * sympy.exp(a * x + b) + d,
#     'poly2': lambda x, a, b, c, d: a * x**2 + b * x + c + d,
#     # ... Add more functions as needed
# }

# Define your symbolic function library 
torch_functions = {
    'sin': lambda x, a, b, c, d: c * torch.sin(a * x + b) + d,
    'exp': lambda x, a, b, c, d: c * torch.exp(a * x + b) + d,
    'poly2': lambda x, a, b, c, d: a * x**2 + b * x + c + d,
    'cos': lambda x, a, b, c, d: c * torch.cos(a * x + b) + d,
    'tan': lambda x, a, b, c, d: c * torch.tan(a * x + b) + d,
#     'log': lambda x, a, b, c, d: c * torch.log(a * x + b) + d,
    'poly3': lambda x, a, b, c, d, e: a * x**3 + b * x**2 + c * x + d + e,
#     'sqrt': lambda x, a, b, c, d: c * torch.sqrt(a * x + b) + d,
    'rational': lambda x, a, b, c, d: (a * x + b) / (c * x + d),
    'gaussian': lambda x, a, b, c, d: a * torch.exp(-((x - b)**2) / (2 * c**2)) + d,
    'logistic': lambda x, a, b, c, d: a / (1 + torch.exp(-b * (x - c))) + d,
    'piecewise': lambda x, a, b, c, d: torch.where(x < b, a * x + c, d * x + c),
    'fourier': lambda x, a, b, c, d, e: a * torch.sin(b * x + c) + d * torch.cos(e * x + c),
}

sympy_functions = {
    'sin': lambda x, a, b, c, d: c * sympy.sin(a * x + b) + d,
    'exp': lambda x, a, b, c, d: c * sympy.exp(a * x + b) + d,
    'poly2': lambda x, a, b, c, d: a * x**2 + b * x + c + d,
    'cos': lambda x, a, b, c, d: c * sympy.cos(a * x + b) + d,
    'tan': lambda x, a, b, c, d: c * sympy.tan(a * x + b) + d,
#     'log': lambda x, a, b, c, d: c * sympy.log(a * x + b) + d,
    'poly3': lambda x, a, b, c, d, e: a * x**3 + b * x**2 + c * x + d + e,
#     'sqrt': lambda x, a, b, c, d: c * sympy.sqrt(a * x + b) + d,
    'rational': lambda x, a, b, c, d: (a * x + b) / (c * x + d),
    'gaussian': lambda x, a, b, c, d: a * sympy.exp(-((x - b)**2) / (2 * c**2)) + d,
    'logistic': lambda x, a, b, c, d: a / (1 + sympy.exp(-b * (x - c))) + d,
    'piecewise': lambda x, a, b, c, d: sympy.Piecewise((a * x + c, x < b), (d * x + c, x >= b)),
    'fourier': lambda x, a, b, c, d, e: a * sympy.sin(b * x + c) + d * sympy.cos(e * x + c),
}
def find_symbolic(x_eval,y_eval,in_feature):
    # Find the best-fitting symbolic function from your library
    best_r2 = 0
    best_func_name = None
    best_params = None
    for func_name, func in torch_functions.items():
        try:
            # Fit the symbolic function to the spline data
            params, _ = curve_fit(func, x_eval, y_eval)
            # Calculate R-squared for the fit
            r2 = torch.sum((y_eval - func(x_eval, *params)) ** 2) / torch.sum(
                (y_eval - torch.mean(y_eval)) ** 2
            )
            if r2 > best_r2:
                best_r2 = r2
                best_func_name = func_name
                best_params = params

        except RuntimeError:
            # Ignore fitting errors (might happen for some functions)
            pass

    if best_func_name is not None:
        # Create a Sympy expression from the symbolic function and fitted params
        symbolic_expression = sympy_functions[best_func_name](
            sympy.Symbol(f"x_{in_feature + 1}"), *best_params
        )
    else:
        # If no good fit is found, append 'None' as a placeholder
        output_formulas.append(None)
        
    return symbolic_expression
        
def solve_layer_formulas(spline_coefficients,grid_points):
    # Initialize an array to store formulas for the current layer
    layer_formulas = []
    print(spline_coefficients.shape)
    # Iterate through the spline coefficients
    for out_feature in range(spline_coefficients.shape[0]):
        # Initialize a list to store the symbolic expressions for the current output
        output_formulas = []
        for in_feature in range(spline_coefficients.shape[1]):
            # Extract coefficients for the current spline
            coeffs = spline_coefficients[out_feature, in_feature, :].cpu().numpy()

            # Extract grid points for the current spline
            knots = grid_points[in_feature, :].cpu().numpy()

            # Create a BSpline object (assuming cubic splines)
            spline_func = BSpline(knots, coeffs, k=3)  

            # Define points to evaluate the spline 
#             print(knots[3], knots[-4])
            x_eval = torch.tensor(np.linspace(knots[3], knots[-4], 100))

            # Evaluate the spline 
            y_eval = torch.tensor(spline_func(x_eval))

            symbolic_expression = find_symbolic(x_eval,y_eval,in_feature)
            
            output_formulas.append(symbolic_expression)
                
        # Now, for the current output, combine the formulas from all input features
        output_formula = 0
        
        for i, formula in enumerate(output_formulas):
            if formula is not None:
                output_formula += formula
        
        layer_formulas.append(output_formula)

    # Append the formulas for the current layer to the overall array
#     best_fit_formulas.append(layer_formulas)
    return layer_formulas

def derive_final_formulas(gformulas):
    # Initialize with the formulas from the first entry
    substitutions = {sympy.Symbol(f"x_{i+1}"): gformulas[0][i] for i in range(len(gformulas[0]))}

    # Iterate through each subsequent layer in gformulas
    for formulas in gformulas[1:]:
        # Update the current layer's formulas with substitutions from previous layers
        updated_formulas = [f.xreplace(substitutions) for f in formulas]

        # Create new substitutions for the next layer
        substitutions = {sympy.Symbol(f"x_{j+1}"): updated_formulas[j] for j in range(len(updated_formulas))}

    # Return the final set of formulas after all substitutions
#     final_formulas = list(substitutions.values())
    return updated_formulas,substitutions 