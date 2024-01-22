from mutation import hyperparameters as mut_hp, gaussian_pertubation
from recombination import hyperparameters as rec_hp, n_point_crossover
from selection import hyperparameters as sel_hp, uniform
import numpy as np

def sepline(text = ""):
    l = "======================" + text + "======================"
    print(l)

def test_hyperparam_change():
    """Test hyperparameter change."""
    sepline("test_hyperparam_change")
    gaussian_pertubation((np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)))

    gaussian_pertubation((np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)))

    print("changing hyperparameters")

    mut_hp["learning_rate_function"] = lambda x: 5.0 / np.sqrt(x)

    gaussian_pertubation((np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)))

def test_gaussian_pertubation():
    """Test Gaussian pertubation."""
    sepline("test_gaussian_pertubation")
    size = 5
    individual = (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, size))
    individual = gaussian_pertubation(individual)
    print(individual)
    assert individual[0].shape == (5,)
    
def test_n_point_crossover():
    sepline("test_n_pos_crossover")
    ret = n_point_crossover(
            (np.array([2361, 2232, 2433, 4236, 2355]), np.random.normal(0, 1, 5)), 
            (np.array([151, 2362, 3235, 4316, 5135]), np.random.normal(0, 1, 5)), 
            n=2)
    print(ret)
    
def test_uniform():
    sepline("test_uniform")
    population = [ 
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
        (np.array([1, 2, 3, 4, 5]), np.random.normal(0, 1, 5)),
    ]
    ret = uniform(population, 5)
    print(ret)
    assert len(ret) == 5
    
    
    
    
def main():
    """Run tests."""
    test_hyperparam_change()
    test_n_point_crossover()
    test_uniform()
    
if __name__ == "__main__":
    main()