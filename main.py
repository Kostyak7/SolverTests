from heat_conduction.heat_cond_tests import test_case_1_dirichlet_bc, test_case_2_neumann_bc, test_case_3_heat_source


def main() -> None:
    print("Running test case 1: Dirichlet boundary conditions")
    test_case_1_dirichlet_bc()
    
    print("\nRunning test case 2: Neumann boundary conditions")
    # test_case_2_neumann_bc()
    
    print("\nRunning test case 3: Heat source")
    # test_case_3_heat_source()


if __name__ == "__main__":
    main()