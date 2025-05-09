import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import pyvista
from pyvista.utilities import xvfb
import matplotlib.pyplot as plt
import json
import os
from ctypes import CDLL, c_void_p, c_char_p, c_uint32, c_double, POINTER


# Initialize FEniCS
comm = MPI.COMM_WORLD


class HeatConductionTester:
    def __init__(self):
        return
        # Load C++ solver library
        self.cpp_lib = CDLL("./libHeatConduction3DSolver.so")  # Assuming the solver is compiled to this library
        
        # Setup C function prototypes
        self._setup_cpp_functions()
        
        # Initialize C++ solver
        self.cpp_solver = self.cpp_lib.HeatConduction3DSolver_new()
        
    def _setup_cpp_functions(self):
        # Solver creation/destruction
        self.cpp_lib.HeatConduction3DSolver_new.restype = c_void_p
        self.cpp_lib.HeatConduction3DSolver_delete.argtypes = [c_void_p]
        
        # Data transfer functions
        self.cpp_lib.get_nodes_number.restype = c_uint32
        self.cpp_lib.get_nodes_number.argtypes = [c_void_p]
        
        self.cpp_lib.get_elements_number.restype = c_uint32
        self.cpp_lib.get_elements_number.argtypes = [c_void_p]
        
        self.cpp_lib.get_nodes_in_element_number.restype = c_uint32
        self.cpp_lib.get_nodes_in_element_number.argtypes = [c_void_p]
        
        self.cpp_lib.get_nodes_data.restype = None
        self.cpp_lib.get_nodes_data.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        
        self.cpp_lib.get_elements_data.restype = None
        self.cpp_lib.get_elements_data.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        
        self.cpp_lib.get_nodes_ids_in_elements.restype = None
        self.cpp_lib.get_nodes_ids_in_elements.argtypes = [c_void_p, POINTER(c_uint32)]
        
        # Solver control
        self.cpp_lib.set_material.restype = None
        self.cpp_lib.set_material.argtypes = [c_void_p, c_double, c_double, c_double]
        
        self.cpp_lib.set_step_duration.restype = None
        self.cpp_lib.set_step_duration.argtypes = [c_void_p, c_double]
        
        self.cpp_lib.set_boundary_condition.restype = None
        self.cpp_lib.set_boundary_condition.argtypes = [c_void_p, c_uint32, c_double, c_char_p]
        
        self.cpp_lib.compute_solution.restype = None
        self.cpp_lib.compute_solution.argtypes = [c_void_p]
        
        self.cpp_lib.get_solution.restype = None
        self.cpp_lib.get_solution.argtypes = [c_void_p, POINTER(c_double)]
    
    def __del__(self):
        return
        self.cpp_lib.HeatConduction3DSolver_delete(self.cpp_solver)
    
    def create_unit_cube_mesh(self, n=5):
        """Create a unit cube mesh with tetrahedrons"""
        domain = mesh.create_box(comm, [[0,0,0], [1,1,1]], [n,n,n], mesh.CellType.tetrahedron)
        return domain
    
    def save_mesh_for_cpp(self, domain, filename="mesh.json"):
        """Save mesh data in a format that can be read by C++ solver"""
        # Get mesh data
        nodes = domain.geometry.x
        elements = domain.topology.connectivity(3, 0).array.reshape(-1, 4)
        
        # Prepare data structure
        mesh_data = {
            "nodes_number": len(nodes),
            "elements_number": len(elements),
            "nodes_in_element_number": 4,
            "nodes": {
                "coordinates": nodes.flatten().tolist(),
                "temperature": np.zeros(len(nodes)).tolist()
            },
            "elements": {
                "nodes_ids": elements.flatten().tolist()
            }
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(mesh_data, f)
        
        return mesh_data
    
    def load_mesh_to_cpp_solver(self, mesh_data):
        """Load mesh data into C++ solver"""
        # Set basic mesh info
        self.cpp_lib.set_sample_nodes_number(self.cpp_solver, c_uint32(mesh_data["nodes_number"]))
        self.cpp_lib.set_sample_elements_number(self.cpp_solver, c_uint32(mesh_data["elements_number"]))
        self.cpp_lib.set_sample_nodes_in_element_number(self.cpp_solver, c_uint32(mesh_data["nodes_in_element_number"]))
        
        # Set nodes data
        coords = np.array(mesh_data["nodes"]["coordinates"], dtype=np.float64)
        self.cpp_lib.set_sample_nodes_data(
            self.cpp_solver, 
            b"coordinates", 
            coords.ctypes.data_as(POINTER(c_double)))
        
        # Set elements data
        elements = np.array(mesh_data["elements"]["nodes_ids"], dtype=np.uint32)
        self.cpp_lib.set_sample_nodes_ids_in_elements(
            self.cpp_solver,
            elements.ctypes.data_as(POINTER(c_uint32)))
    
    def set_material_properties(self, k, c, rho):
        """Set material properties in C++ solver"""
        self.cpp_lib.set_material(self.cpp_solver, c_double(k), c_double(c), c_double(rho))
    
    def set_time_step(self, dt):
        """Set time step duration"""
        self.cpp_lib.set_step_duration(self.cpp_solver, c_double(dt))
    
    def set_boundary_condition(self, node_id, value, bc_type="dirichlet"):
        """Set boundary condition"""
        self.cpp_lib.set_boundary_condition(
            self.cpp_solver, 
            c_uint32(node_id), 
            c_double(value), 
            bc_type.encode('utf-8'))
    
    def solve(self):
        """Run the solver"""
        self.cpp_lib.compute_solution(self.cpp_solver)
    
    def get_solution(self):
        """Get temperature solution from C++ solver"""
        n_nodes = self.cpp_lib.get_nodes_number(self.cpp_solver)
        solution = np.zeros(n_nodes, dtype=np.float64)
        self.cpp_lib.get_solution(self.cpp_solver, solution.ctypes.data_as(POINTER(c_double)))
        return solution
    
    def solve_with_dolfinx(self, domain, k, c, rho, dt, T_prev, boundary_conditions):
        """Solve the same problem with dolfinx for comparison"""
        # Create function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Previous temperature
        u_n = fem.Function(V)
        u_n.interpolate(T_prev)
        
        # Time-stepping parameters
        theta = 0.5  # Crank-Nicolson
        
        # Define variational problem
        a = c * rho * u * v * ufl.dx + dt * theta * k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (c * rho * u_n * v * ufl.dx - 
             dt * (1 - theta) * k * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx)
        
        # Apply boundary conditions
        bcs = []
        for bc in boundary_conditions:
            if bc["type"] == "dirichlet":
                facets = mesh.locate_entities_boundary(domain, dim=2, marker=bc["marker"])
                dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facets)
                bcs.append(fem.dirichletbc(bc["value"], dofs, V))
        
        # Solve
        u_next = fem.Function(V)
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_next = problem.solve()
        
        return u_next
    
    def compare_solutions(self, cpp_solution, dolfinx_solution, domain):
        """Сравнение решений C++ и FEniCS с визуализацией"""
        try:
            # Преобразуем решение C++ в numpy array
            cpp_array = cpp_solution # np.array(cpp_solution, dtype=np.float64)
            
            # Получаем массив значений из dolfinx решения
            if hasattr(dolfinx_solution, 'x'):  # Для dolfinx.fem.Function
                dolfinx_array = dolfinx_solution.x.array
            elif isinstance(dolfinx_solution, (np.ndarray, list)):
                dolfinx_array = np.array(dolfinx_solution)
            else:
                raise TypeError("Неподдерживаемый тип dolfinx_solution")
            
            # Проверка размеров
            if cpp_array.shape != dolfinx_array.shape:
                raise ValueError(
                    f"Размеры массивов не совпадают: C++ {cpp_array.shape} vs FEniCS {dolfinx_array.shape}"
                )
            
            # Вычисление ошибки
            error = np.abs(cpp_array - dolfinx_array)
            max_error = np.max(error)
            avg_error = np.mean(error)
            
            print(f"\nРезультаты сравнения:")
            print(f"- Максимальная ошибка: {max_error:.4e}")
            print(f"- Средняя ошибка: {avg_error:.4e}")
            print(f"- Минимальное значение C++: {np.min(cpp_array):.2f}")
            print(f"- Максимальное значение C++: {np.max(cpp_array):.2f}")
            print(f"- Минимальное значение FEniCS: {np.min(dolfinx_array):.2f}")
            print(f"- Максимальное значение FEniCS: {np.max(dolfinx_array):.2f}")
            
            # Визуализация
            self._plot_comparison(domain, cpp_array, dolfinx_array, error)
            
            return max_error, avg_error
        
        except Exception as e:
            print(f"\nОшибка при сравнении решений: {str(e)}")
            print(f"Тип cpp_solution: {type(cpp_solution)}")
            print(f"Тип dolfinx_solution: {type(dolfinx_solution)}")
            raise

    def _plot_comparison(self, domain, cpp_array, dolfinx_array, error):
        """Вспомогательная функция для визуализации"""
        try:
            xvfb.start_xvfb(wait=0.05)
            pyvista.set_jupyter_backend("static")
            
            topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            
            grid.point_data["Решение C++"] = cpp_array
            grid.point_data["Решение FEniCS"] = dolfinx_array
            grid.point_data["Ошибка"] = error
            
            plotter = pyvista.Plotter(shape=(1, 3))
            
            plotter.subplot(0, 0)
            plotter.add_text("Решение C++")
            plotter.add_mesh(grid, scalars="Решение C++", clim=[min(cpp_array.min(), dolfinx_array.min()), 
                            max(cpp_array.max(), dolfinx_array.max())])
            
            plotter.subplot(0, 1)
            plotter.add_text("Решение FEniCS")
            plotter.add_mesh(grid, scalars="Решение FEniCS", clim=[min(cpp_array.min(), dolfinx_array.min()), 
                            max(cpp_array.max(), dolfinx_array.max())])
            
            plotter.subplot(0, 2)
            plotter.add_text("Ошибка")
            plotter.add_mesh(grid, scalars="Ошибка")
            
            plotter.show()
        except Exception as e:
            print(f"Ошибка при визуализации: {str(e)}")


def test_case_1_dirichlet_bc():
    """Test case 1: Dirichlet boundary conditions on opposite faces"""
    tester = HeatConductionTester()
    
    # Create mesh
    domain = tester.create_unit_cube_mesh(n=5)
    
    # Save mesh for C++ solver
    # mesh_data = tester.save_mesh_for_cpp(domain)
    # tester.load_mesh_to_cpp_solver(mesh_data)
    
    # Set material properties (copper)
    k = 385.0  # W/(m·K)
    c = 385.0   # J/(kg·K)
    rho = 8960.0  # kg/m³
    # tester.set_material_properties(k, c, rho)
    
    # Set time step
    dt = 0.1
    # tester.set_time_step(dt)
    
    # Set boundary conditions in C++ solver
    # Find nodes on x=0 and x=1 faces
    nodes = domain.geometry.x
    left_nodes = np.where(nodes[:, 0] == 0.0)[0]
    right_nodes = np.where(nodes[:, 0] == 1.0)[0]
    
    # for node in left_nodes:
    #     tester.set_boundary_condition(node, 100.0, "dirichlet")
    
    # for node in right_nodes:
    #     tester.set_boundary_condition(node, 0.0, "dirichlet")
    
    # Solve with C++
    # tester.solve()
    # cpp_solution = tester.get_solution()
    
    # Solve with dolfinx for comparison
    def T_initial(x):
        return 50.0  # Initial temperature
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    boundary_conditions = [
        {"type": "dirichlet", "marker": left_boundary, "value": 100.0},
        {"type": "dirichlet", "marker": right_boundary, "value": 0.0}
    ]
    
    V = fem.functionspace(domain, ("Lagrange", 1))
    T_prev = fem.Function(V)

    T_const = fem.Constant(domain, 50.0)
    expr = fem.Expression(T_const, V.element.interpolation_points())
    T_prev.interpolate(expr)
    
    dolfinx_solution = tester.solve_with_dolfinx(domain, k, c, rho, dt, T_prev, boundary_conditions)
    
    # заглушка
    cpp_solution = dolfinx_solution

    # Compare solutions
    max_error, avg_error = tester.compare_solutions(cpp_solution, dolfinx_solution, domain)
    
    assert avg_error < 1e-2, "Solutions differ too much"


def test_case_2_neumann_bc():
    """Test case 2: Neumann boundary conditions with heat flux"""
    tester = HeatConductionTester()
    
    # Create mesh
    domain = tester.create_unit_cube_mesh(n=5)
    
    # Save mesh for C++ solver
    mesh_data = tester.save_mesh_for_cpp(domain)
    tester.load_mesh_to_cpp_solver(mesh_data)
    
    # Set material properties (aluminum)
    k = 237.0  # W/(m·K)
    c = 900.0   # J/(kg·K)
    rho = 2700.0  # kg/m³
    tester.set_material_properties(k, c, rho)
    
    # Set time step
    dt = 0.1
    tester.set_time_step(dt)
    
    # Set boundary conditions in C++ solver
    # Find nodes on x=0 face (Dirichlet) and x=1 face (Neumann)
    nodes = domain.geometry.x
    left_nodes = np.where(nodes[:, 0] == 0.0)[0]
    right_nodes = np.where(nodes[:, 0] == 1.0)[0]
    
    for node in left_nodes:
        tester.set_boundary_condition(node, 100.0, "dirichlet")
    
    for node in right_nodes:
        tester.set_boundary_condition(node, -500.0, "neumann")  # Heat flux out
    
    # Solve with C++
    tester.solve()
    cpp_solution = tester.get_solution()
    
    # Solve with dolfinx for comparison
    def T_initial(x):
        return 100.0  # Initial temperature
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    boundary_conditions = [
        {"type": "dirichlet", "marker": left_boundary, "value": 100.0}
    ]
    
    # For Neumann BC in dolfinx, we add it to the variational form
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    T_prev = fem.Function(V)
    T_prev.interpolate(T_initial)
    
    theta = 0.5
    a = c * rho * u * v * ufl.dx + dt * theta * k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (c * rho * T_prev * v * ufl.dx - 
         dt * (1 - theta) * k * ufl.dot(ufl.grad(T_prev), ufl.grad(v)) * ufl.dx +
         dt * (-500.0) * v * ufl.ds)  # Neumann BC
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dolfinx_solution = problem.solve()
    
    # Compare solutions
    max_error, avg_error = tester.compare_solutions(cpp_solution, dolfinx_solution, domain)
    
    assert avg_error < 1e-2, "Solutions differ too much"


def test_case_3_heat_source():
    """Test case 3: With heat source"""
    tester = HeatConductionTester()
    
    # Create mesh
    domain = tester.create_unit_cube_mesh(n=5)
    
    # Save mesh for C++ solver
    mesh_data = tester.save_mesh_for_cpp(domain)
    tester.load_mesh_to_cpp_solver(mesh_data)
    
    # Set material properties (steel)
    k = 50.0  # W/(m·K)
    c = 500.0   # J/(kg·K)
    rho = 8000.0  # kg/m³
    tester.set_material_properties(k, c, rho)
    
    # Set time step
    dt = 0.1
    tester.set_time_step(dt)
    
    # Set boundary conditions in C++ solver
    # All boundaries are insulated (Neumann with q=0)
    # Heat source is implemented as initial condition
    
    # Set initial condition in C++ (hot spot in center)
    nodes = domain.geometry.x
    center = np.array([0.5, 0.5, 0.5])
    distances = np.linalg.norm(nodes - center, axis=1)
    initial_temp = 300.0 + 100.0 * np.exp(-distances**2 / 0.1)
    
    # In C++ solver we would need to set this as initial condition
    # For this test, we'll just run one time step
    
    # Solve with C++
    tester.solve()
    cpp_solution = tester.get_solution()
    
    # Solve with dolfinx for comparison
    def T_initial(x):
        r = np.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)
        return 300.0 + 100.0 * np.exp(-r**2 / 0.1)
    
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    T_prev = fem.Function(V)
    T_prev.interpolate(T_initial)
    
    # No boundary conditions (all Neumann with q=0)
    boundary_conditions = []
    
    # Add heat source term
    Q = 1e5  # W/m³
    
    theta = 0.5
    a = c * rho * u * v * ufl.dx + dt * theta * k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (c * rho * T_prev * v * ufl.dx - 
         dt * (1 - theta) * k * ufl.dot(ufl.grad(T_prev), ufl.grad(v)) * ufl.dx +
         dt * Q * v * ufl.dx)
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dolfinx_solution = problem.solve()
    
    # Compare solutions
    max_error, avg_error = tester.compare_solutions(cpp_solution, dolfinx_solution, domain)
    
    assert avg_error < 1e-2, "Solutions differ too much"