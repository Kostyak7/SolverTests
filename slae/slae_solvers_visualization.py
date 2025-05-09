import matplotlib.pyplot as plt
import numpy as np

linestyles = {
        "Gauss": "-",       
        "BiCGSTAB": "-",    
        "FGMRES": "-",      
        "BiCGSTAB+ILU": "-",
        "FGMRES+ILU": "-"   
}

linecolors = {
    "Gauss": "red",       
    "BiCGSTAB": "g",    
    "FGMRES": "c",      
    "BiCGSTAB+ILU": "orange",
    "FGMRES+ILU": "blue"   
}

solvers_to_show = {
    "Gauss", 
    "BiCGSTAB", 
    "CG", 
    "FGMRES", 
    "GMRES", 
    "MINRES", 
    # "BiCGSTAB+ILU",
    # "FGMRES+ILU",
    # "GMRES+ILU",
    # "BiCGSTAB+MG", 
    # "CG+MG", 
    # "FGMRES+MG", 
    # "GMRES+MG", 
    # "MINRES+MG",  
}

def parse_file_to_dict(filename: str, allowed_keys: set) -> dict:
    result = {}
    current_solver = None
    file_path = 'data/' +  filename + '.txt'
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Solver:'):
                current_solver = line.split('Solver:')[1].strip()
                if current_solver in allowed_keys:
                    result[current_solver] = []
            elif line.startswith('time:') or line == 'time:':
                continue
            elif line and current_solver is not None and current_solver in allowed_keys:
                times = [t.strip() for t in line.split(',') if t.strip()]
                times_float = [float(t.replace(',', '')) for t in times]
                result[current_solver].extend(times_float)
    
    return result


def create_plot(y_values, solvers, xlabel, ylabel, islog=False, title='', linecolors=linecolors, linestyles=linestyles):
    plt.figure(figsize=(12, 8))
    for name, values in solvers.items():
        plt.plot(y_values, values, marker='o', label=name, linestyle=linestyles.get(name, "-")) # , color=linecolors[name])

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    if islog:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel + ' (логарифмическая шкала)')
        plt.ylabel(ylabel + ' (логарифмическая шкала)')
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if len(title):
        plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


def plot_1(islog=False):
    y_values = [100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    solvers = parse_file_to_dict('diff_sizes_test', solvers_to_show)

    create_plot(y_values, solvers, 'Размер задачи', 'Время выполнения (мс)', islog)


def plot_2(islog=False):
    y_values = [10, 25, 50, 75, 100, 150, 300, 500]
    solvers = parse_file_to_dict('diff_isl_test', solvers_to_show)

    create_plot(y_values, solvers, 'Ширина ленты', 'Время выполнения (мс)', islog)


def plot_3(islog=False):
    y_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    solvers = parse_file_to_dict('diff_sparsity_test', solvers_to_show)

    create_plot(y_values, solvers, 'Частота нулей в ленте', 'Время выполнения (мс)', islog)


if __name__ == "__main__":
    plot_1(False)
    plot_2(False)
    plot_3(False)
