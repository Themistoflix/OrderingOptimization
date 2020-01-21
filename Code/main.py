import Code.Nanoparticle as NP
import Code.FCCLattice as FCC
import Code.CuttingPlaneUtilities as CPU
import Code.GlobalFeatureClassifier as GFC

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import BayesianRidge

import copy


def findClusters(particles, numberOfClusters, l_max,  computeBondParameters=True):
    bondParameters = list()
    for step, particle in enumerate(particles):
        print("Finding Clusters. Step: {0}".format(step))

        if computeBondParameters:
            print(l_max)
            particle.computeBondParameters(l_max)
        bondParameters = bondParameters + list(particle.getBondParameters().values())

    kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(bondParameters)
    return kmeans


def ridgeRegression(particles, kmeans, recomputeEnergies=True):
    energies = list()
    featureVectors = list()

    for step, particle in enumerate(particles):
        print("Computing Energies for Bayesian Ridge Regression. Step: {0}".format(step))
        if recomputeEnergies:
            particle.computeEMTEnergy(50)
        particle.compute_feature_vector(kmeans)

        energies.append(particle.getEMTEnergy())
        featureVectors.append(particle.getFeatureVector())

    energies = np.array(energies)
    ridge = BayesianRidge(fit_intercept=False)
    ridge.fit(featureVectors, energies)

    return ridge


def runAutobagSimulation(N_steps, startPopulation, kmeans, ridge,
                         startParticle, randomOperations):
    totalPopulation = startPopulation

    startParticle.compute_feature_vector(kmeans, l_max, True)
    startParticle.computeEMTEnergy(50)
    startParticle.computeRREnergy(ridge)

    acceptedParticles = [startParticle]
    emtEnergies = [startParticle.getEMTEnergy()]
    rrEnergies = [startParticle.getRREnergy()]

    rattleOperator = SRO.SurfaceRattleOperator(ridge, kmeans)
    for step in range(N_steps):
        print("__________________________________")
        print("Step: {0}".format(step))
        print("__________________________________")

        if not randomOperations:
            newParticle = rattleOperator.magneticRattle(startParticle)
        else:
            newParticle = rattleOperator.randomRattle(startParticle)

        newParticle.compute_feature_vector(kmeans, True)

        newParticle.computeEMTEnergy(50)
        newParticle.computeRREnergy(ridge)
        emtEnergies.append(newParticle.getEMTEnergy())
        rrEnergies.append(newParticle.getRREnergy())
        totalPopulation.append(newParticle)

        print('RR energy: {0}'.format(newParticle.getRREnergy()))

        allFeatures = [particle.getFeatureVector() for particle in totalPopulation]
        allEnergies = np.array([particle.getEMTEnergy() for particle in totalPopulation])
        ridge.fit(allFeatures, allEnergies)

        beta = 4
        deltaE = newParticle.getEMTEnergy() - startParticle.getEMTEnergy()
        if deltaE < 0:
            acceptanceRate = 1
        else:
            acceptanceRate = np.exp(-beta * deltaE)

        if np.random.random() < acceptanceRate:
            print('accepted at: {0}'.format(newParticle.getEMTEnergy()))
            startParticle = newParticle

            acceptedParticles.append(newParticle)

    return acceptedParticles, emtEnergies, rrEnergies


if __name__ == '__main__':
    lattice = FCC.FCCLattice(15, 15, 15, 2)
    cutting_plane_generator = CPU.SphericalCuttingPlaneGenerator(3., 11.)

    training_set = list()
    simple_classifier = GFC.SimpleFeatureClassifier()

    population_size = 20
    for i in range(population_size):
        particle = NP.Nanoparticle(lattice, 5)
        particle.convex_shape([42, 43], ['Pt', 'Au'], 9, 9, 9, cutting_plane_generator)
        #simple_classifier.compute_feature_vector(particle)

        particle.optimize_coordination_numbers()
        particle.computeEMTEnergy(steps=20)
        simple_classifier.compute_feature_vector(particle)
        training_set.append(particle)


