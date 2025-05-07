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
    # "BiCGSTAB", 
    # "CG", 
    # "FGMRES", 
    # "GMRES", 
    # "MINRES", 
    # "BiCGSTAB+JACOBI",
    # "CG+JACOBI", 
    # "FGMRES+JACOBI", 
    # "GMRES+JACOBI",
    # "MINRES+JACOBI",
    "BiCGSTAB+ILU",
    # "FGMRES+ILU",
    # "MINRES+ILU",
    # "BiCGSTAB+MG", 
    # "CG+MG", 
    # "FGMRES+MG", 
    # "GMRES+MG", 
    # "MINRES+MG",  
}

def parse_file_to_dict(file_path: str, allowed_keys: set) -> dict:
    result = {}
    current_solver = None
    
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
    file_path = 'diff_sizes_test.txt'
    y_values = [100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    solvers = parse_file_to_dict(file_path, solvers_to_show)
    # solvers = {
    #     "Gauss": [2.979, 21.313, 70.067, 163.121, 316.942, 2500.5, 8405.65, 19898.2, 67343.7, 159747, 312475],
    #     "BiCGSTAB": [115.84, 933.9, 3932.13, 7016.27, 11053.2, 45472.7, 104268, 189768, 460076, 864274, 1.37938e+06],
    #     "CG": [33.979, 297.738, 1088.93, 2596.03, 5171.33, 22738, 52182.4, 94768.7, 230357, 432539, 689500],
    #     "FGMRES": [209.937, 1014.65, 2184.92, 3828.59, 5972.78, 23951.3, 54522.9, 98698.5, 238653, 448281, 714077],
    #     "GMRES": [208.975, 1013.64, 2181.44, 3819.88, 5960.96, 23969.4, 54660.8, 98812.1, 240141, 447989, 715519],
    #     "MINRES": [172.799, 907.688, 2006.78, 3568.85, 5596.62, 22894.6, 52442, 94981.6, 229940, 432089, 692425],
    #     "BiCGSTAB+JACOBI": [84.359, 184.773, 493.068, 6314.29, 3575.55, 26799.4, 77485.5, 190233, 459191, 863589, 1.37776e+06],
    #     "CG+JACOBI": [157.611, 884.338, 1970.92, 3514.08, 5554.26, 22773.4, 52203, 94729.8, 229342, 431132, 690107],
    #     "FGMRES+JACOBI": [209.66, 1023.28, 2177.6, 3818.51, 5959.21, 24040.1, 54659.8, 98829.5, 238971, 447849, 715129],
    #     "GMRES+JACOBI": [211.245, 1019.73, 2182.28, 3814.04, 5958.9, 23981, 54620.9, 98878.1, 238285, 448469, 717536],
    #     "MINRES+JACOBI": [172.105, 908.074, 2008.24, 3555.16, 5602.21, 22905.3, 52420.4, 94985.8, 230555, 434278, 691862],
    #     "BiCGSTAB+ILU": [2.422, 13.765, 40.882, 91.989, 175.871, 1376.98, 4628.38, 11000.7, 40595.5, 109327, 234993],
    #     "FGMRES+ILU": [262.849, 1310.23, 2861.58, 5044.52, 7923.48, 32698.8, 76248, 0, 344932, 0, 0],
    #     "MINRES+ILU": [174.076, 928.925, 2033.09, 3632.9, 5743.73, 24207.4, 56799.1, 105629, 270366, 540800, 927602]
    # }

    create_plot(y_values, solvers, 'Размер задачи', 'Время выполнения (мс)', islog)


def plot_2(islog=False):
    file_path = 'diff_isl_test.txt'  
    y_values = [10, 25, 50, 75, 100, 150, 300, 500]
    solvers = parse_file_to_dict(file_path, solvers_to_show)

    create_plot(y_values, solvers, 'Ширина ленты', 'Время выполнения (мс)', islog)


def plot_3(islog=False):
    file_path = 'diff_sparsity_test.txt'  
    y_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    solvers = parse_file_to_dict(file_path, solvers_to_show)

    create_plot(y_values, solvers, 'Частота нулей в ленте', 'Время выполнения (мс)', islog)


if __name__ == "__main__":
    plot_2(True)
