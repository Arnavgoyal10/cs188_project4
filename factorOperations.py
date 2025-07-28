# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined


def joinFactorsByVariableWithCallTracking(callTrackingList=None):

    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that
        contain that variable.

        Returns a tuple of
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(("join", joinVariable))

        currentFactorsToJoin = [
            factor for factor in factors if joinVariable in factor.variablesSet()
        ]
        currentFactorsNotToJoin = [
            factor for factor in factors if joinVariable not in factor.variablesSet()
        ]

        # typecheck portion
        numVariableOnLeft = len(
            [
                factor
                for factor in currentFactorsToJoin
                if joinVariable in factor.unconditionedVariables()
            ]
        )
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", joinVariable)
            raise ValueError(
                "The joinBy variable can only appear in one factor as an \nunconditioned variable. \n"
                + "joinVariable: "
                + str(joinVariable)
                + "\n"
                + ", ".join(
                    map(
                        str,
                        [
                            factor.unconditionedVariables()
                            for factor in currentFactorsToJoin
                        ],
                    )
                )
            )

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########


def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.

    You should calculate the set of unconditioned variables and conditioned
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input
    (such as getProbability and setProbability) can handle
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError(
                "unconditionedVariables can only appear in one factor. \n"
                + "unconditionedVariables: "
                + str(intersect)
                + "\nappear in more than one input factor.\n"
                + "Input factors: \n"
                + "\n".join(map(str, factors))
            )

    "*** YOUR CODE HERE ***"
    if not factors:
        return None

    # Convert to list if it's a dict_values object
    factors_list = (
        list(factors)
        if hasattr(factors, "__iter__") and not isinstance(factors, list)
        else factors
    )

    if not factors_list:
        return None

    # Get the variable domains from the first factor (all factors have the same domains)
    variableDomainsDict = factors_list[0].variableDomainsDict()

    # Calculate the union of unconditioned and conditioned variables
    all_unconditioned = set()
    all_conditioned = set()

    for factor in factors_list:
        all_unconditioned.update(factor.unconditionedVariables())
        all_conditioned.update(factor.conditionedVariables())

    # A variable is unconditioned in the result if it appears as unconditioned in ANY factor
    # A variable is conditioned in the result if it appears as conditioned in ANY factor AND is not unconditioned in any factor
    result_unconditioned = all_unconditioned
    result_conditioned = all_conditioned - all_unconditioned

    # Create the new factor
    result_factor = Factor(
        list(result_unconditioned), list(result_conditioned), variableDomainsDict
    )

    # For each possible assignment, compute the product of probabilities from all factors
    for assignment in result_factor.getAllPossibleAssignmentDicts():
        product = 1.0
        for factor in factors_list:
            product *= factor.getProbability(assignment)
        result_factor.setProbability(assignment, product)

    return result_factor
    "*** END YOUR CODE HERE ***"


########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        You should calculate the set of unconditioned variables and conditioned
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(("eliminate", eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError(
                "Elimination variable is not an unconditioned variable "
                + "in this factor\n"
                + "eliminationVariable: "
                + str(eliminationVariable)
                + "\nunconditionedVariables:"
                + str(factor.unconditionedVariables())
            )

        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError(
                "Factor has only one unconditioned variable, so you "
                + "can't eliminate \nthat variable.\n"
                + "eliminationVariable:"
                + str(eliminationVariable)
                + "\n"
                + "unconditionedVariables: "
                + str(factor.unconditionedVariables())
            )

        "*** YOUR CODE HERE ***"
        # Get the variable domains
        variableDomainsDict = factor.variableDomainsDict()

        # Calculate the new unconditioned and conditioned variables
        new_unconditioned = list(
            factor.unconditionedVariables() - {eliminationVariable}
        )
        new_conditioned = list(factor.conditionedVariables())

        # Create the new factor
        result_factor = Factor(new_unconditioned, new_conditioned, variableDomainsDict)

        # For each possible assignment in the new factor, sum over all values of the elimination variable
        for assignment in result_factor.getAllPossibleAssignmentDicts():
            total_prob = 0.0

            # Sum over all possible values of the elimination variable
            for value in variableDomainsDict[eliminationVariable]:
                # Create assignment including the elimination variable
                full_assignment = assignment.copy()
                full_assignment[eliminationVariable] = value

                # Add the probability from the original factor
                total_prob += factor.getProbability(full_assignment)

            # Set the summed probability in the result factor
            result_factor.setProbability(assignment, total_prob)

        return result_factor
        "*** END YOUR CODE HERE ***"

    return eliminate


eliminate = eliminateWithCallTracking()
