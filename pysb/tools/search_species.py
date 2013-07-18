## PySB module to search for monomers in lists of species in a model
##
## Alex Lubbock <alex.lubbock@ed.ac.uk>
## 3rd July 2013

import pysb
from warnings import warn

def remove_duplicates_by_eq(obj_list):
    """
    Remove duplicates from a list using equality operator (inefficient)
    """
    new_list = []
    for i in obj_list:
        if i not in new_list:
            new_list.append(i)
    return new_list

def complex_patterns_to_monomer_patterns(complex_patterns):
    """
    Extract a list of MonomerPattern objects from a list of ComplexPattern objects.
    Note that each ComplexPattern can contain multiple MonomerPatterns, so the returned
    list may be longer than the list supplied as an argument.
    """
    # Check argument
    if type(complex_patterns) != list or any([type(cp)!=pysb.core.ComplexPattern for cp in complex_patterns]):
        raise TypeError('Expecting a list of ComplexPattern objects')

    # Extract the monomers as a list
    return [m for mlist in complex_patterns for m in mlist.monomer_patterns]

def observable_monomer_patterns(model):
    """
    Returns a list of observable MonomerPatterns from a model
    """
    # Check argument
    if type(model) != pysb.core.Model:
        raise TypeError('Argument should be a PySB model')

    # Get ReactionPatterns
    obs_reactions = [obs.reaction_pattern for obs in model.observables.values()]
    # Extract ComplexPatterns from ReactionPatterns
    obs_complex_patterns = [cp for r in obs_reactions for cp in r.complex_patterns]
    # Extract Monomers from ComplexPatterns
    return remove_duplicates_by_eq(complex_patterns_to_monomer_patterns(obs_complex_patterns))

def initial_condition_monomer_patterns(model):
    """
    Return a list of initial condition MonomerPatterns from a model
    """
    # Check argument
    if type(model) != pysb.core.Model:
        raise TypeError('Argument should be a PySB model')

    # Get initial condition ComplexPatterns
    ic_complex_patterns = [ic[0] for ic in model.initial_conditions]
    # Extract Monomers from ComplexPatterns
    return remove_duplicates_by_eq(complex_patterns_to_monomer_patterns(ic_complex_patterns))

def find_monomers_in_species(monomer_list, model, verbose=True):
    """
    Given a list of MonomerPatterns, returns a list of species which
    contain those patterns
    """
    
    # Check arguments
    if type(monomer_list) != list or any([type(m)!=pysb.core.MonomerPattern for m in monomer_list]):
        raise TypeError('monomer_list should be a list of MonomerPattern objects')
    if type(model) != pysb.core.Model:
        raise TypeError('model should be a PySB model')

    if len(model.species)==0:
        warn('Model species list is empty. Perhaps you need to generate equations?')

    species_ids = []
    for i,m in enumerate(monomer_list):
        # get the species IDs of MonomerPatterns which are
        # in the species list by matching the Monomer objects
        s_ids = [si for si,s in enumerate(model.species) if m.monomer in [sp.monomer for sp in s.monomer_patterns]]
        # Check that the matched MonomerPatterns have site_conditions
        # which are a subset of those in the species list MonomerPatterns
        s_ids = [si for si in s_ids if any([True for mp in model.species[si].monomer_patterns if m.monomer == mp.monomer and all(item in mp.site_conditions.items() or item[1] == pysb.ANY for item in m.site_conditions.items())])]
        # Save the matching IDs
        species_ids.extend(s_ids)
        if verbose:
            print '%d %s found in species %s' % (i,str(m),s_ids)

    return [s for i,s in enumerate(model.species) if i in species_ids]

if __name__ == "__main__":
    import pysb.examples.simple_egfr as egfr
    import pysb.bng

    model = egfr.model
    pysb.bng.generate_equations(model)
    obs_mon = observable_monomer_patterns(model)
    matching_species = find_monomers_in_species(obs_mon, model)
    
    print("Searching for the following observables: ")
    print("\n".join(str(m) for m in obs_mon))
    print("\nMatching species:")
    print("\n".join(str(s) for s in matching_species))
