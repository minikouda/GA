import pytest
import numpy as np
from GA.GA import GeneticSelector

class TestGeneticSelector:
    def setup_method(self):
        # Create dummy data
        self.X = np.random.rand(20, 10)
        self.y = np.random.rand(20)
        self.ga = GeneticSelector(self.X, self.y, pop_size=10, n_gen=5, n_splits=2)

    def test_initialization(self):
        assert self.ga.population.shape == (10, 10)
        assert np.all((self.ga.population == 0) | (self.ga.population == 1))

    def test_fitness(self):
        # Test fitness of a random individual
        ind = self.ga.population[0]
        fit, r2 = self.ga.fitness(ind)
        assert isinstance(fit, float)
        assert isinstance(r2, float)
        
        # Test empty individual
        empty_ind = np.zeros(10)
        fit, r2 = self.ga.fitness(empty_ind)
        assert fit == -1.0
        assert r2 == -1.0

    def test_evaluate_population(self):
        fits, r2s = self.ga.evaluate_population()
        assert len(fits) == 10
        assert len(r2s) == 10

    def test_select_parents(self):
        fits = np.random.rand(10)
        parents = self.ga.select_parents(fits)
        assert parents.shape == (10, 10)
        # Parents should be from the population
        # (This is hard to strictly prove without tracking, but shape is good check)

    def test_crossover(self):
        parents = self.ga.population.copy()
        offspring = self.ga.crossover(parents)
        assert offspring.shape == (10, 10)
        
    def test_mutate(self):
            offspring = self.ga.population.copy()
            pop_size = self.ga.pop_size 
            dummy_fitness = np.full(pop_size, 0.5) 
            dummy_avg_fitness = 0.5
            mutated = self.ga.mutate(offspring, dummy_fitness, dummy_avg_fitness)
            
            assert mutated.shape == (10, 10)
            assert np.all((mutated == 0) | (mutated == 1))

    def test_run(self):
        best_ind, best_r2, best_fit = self.ga.run()
        assert best_ind.shape == (10,)
        assert isinstance(best_r2, float)
        assert isinstance(best_fit, float)
