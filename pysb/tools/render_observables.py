#!/usr/bin/env python

import pysb
import pysb.bng
import sympy
import re
import sys
import os
import pygraphviz
from search_species import observable_monomer_patterns, find_monomers_in_species
# information flow-related
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from qmbpmn.common.graph.csrgraph import CSRDirectedGraph
from qmbpmn.ITMProbe.commands import model_classes

def run(model, include_monomers=False):
    ## generate the equations
    pysb.bng.generate_equations(model)

    graph = pygraphviz.AGraph(directed=True, rankdir="LR")

    ## extract species involved in initial conditions and observables
    ic_species = [cp for cp, parameter in model.initial_conditions]
    obs_species = [obs.reaction_pattern for obs in model.observables.values()]

    ## extract the ComplexPattern for each observable's ObservablePattern
    ## for the moment, require a 1-to-1 relationship
    ##if not all([len(obs.complex_patterns)==1 for obs in obs_species]):
    ##    raise NotImplementedError('Can only handle ObservablePatterns with one ComplexPattern for now')
    ##obs_species = [obs.complex_patterns[0] for obs in obs_species]

    ## we're only interested in monomeric observables for now
    ##obs_species_monomers = [obs for obs in obs_species if len(obs.monomer_patterns)==1]
    all_obs_s_ids = []
    if include_monomers:
        obs_species_mp = observable_monomer_patterns(model)

    ## check where to find observable monomers in the list of species
        obs_s_ids = []
        for oi,o in enumerate(obs_species_mp):
            ## get the species IDs of observable monomers which are
            ## in the species list
            s_ids = [si for si,s in enumerate(model.species) if o.monomer in [sp.monomer for sp in s.monomer_patterns]]
            # check the sites have identical conditions on both monomers
            #        s_ids = [si for si in s_ids if any([True for mp in model.species[si].monomer_patterns if x.monomer == mp.monomer and x.site_conditions == mp.site_conditions])]
            s_ids = [si for si in s_ids if any([True for mp in model.species[si].monomer_patterns if o.monomer == mp.monomer and all(item in mp.site_conditions.items() or item[1] == pysb.ANY for item in o.site_conditions.items())])]
            all_obs_s_ids.extend(s_ids)
            obs_s_ids.append(s_ids)
            #print '%d %s found in species %s' % (oi,str(o),s_ids)
            monomer_node = 'm%d' % oi
            graph.add_node(monomer_node,
                       label=str(o),
                       shape="Mrecord",
                       fillcolor="#FF7216",
                       style="filled", color="transparent",
                       fontsize="12",
                       margin="0.06,0")

        all_obs_s_ids = set(all_obs_s_ids)

    for i, cp in enumerate(model.species):
        species_node = 's%d' % i
        slabel = re.sub(r'% ', r'%\\l', str(cp))
        slabel += '\\l'
        color = "#ccffcc"
        # color species with an initial condition differently
        if len([s for s in ic_species if s.is_equivalent_to(cp)]):
            color = "#aaffff"
        if i in all_obs_s_ids:
            color = "#ff66ff"
        graph.add_node(species_node,
                       label=slabel,
                       shape="Mrecord",
                       fillcolor=color, style="filled", color="transparent",
                       fontsize="12",
                       margin="0.06,0")
    for i, reaction in enumerate(model.reactions_bidirectional):
#        reaction_node = 'r%d' % i
#        graph.add_node(reaction_node,
#                       label=reaction_node,
#                       shape="circle",
#                       fillcolor="lightgray", style="filled", color="transparent",
#                       fontsize="12",
#                       width=".3", height=".3", margin="0.06,0")
        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        modifiers = reactants & products
        reactants = reactants - modifiers
        products = products - modifiers
        attr_reversible = {'dir': 'both', 'arrowtail': 'empty'} if reaction['reversible'] else {}
        for s1 in reactants:
            for s2 in products:
                s_link(graph, s1, s2, **attr_reversible)
        for s1 in modifiers:
            for s2 in modifiers:
                s_link(graph, s1, s2, {'arrowhead': 'odiamond', 'dir': 'both', 'arrowtail': 'odiamond'})

    ## link the monomers
    if include_monomers:
        for oi,o in enumerate(obs_species_monomers):
            for s in obs_s_ids[oi]:
                m_link(graph, oi, s)
    return graph

def m_link(graph, m, s, **attrs):
    nodes = ('m%d' % m, 's%d' %s)
    attrs.setdefault('arrowhead','normal')
    graph.add_edge(*nodes, **attrs)

def s_link(graph, s1, s2, **attrs):
    nodes = ('s%d' % s1, 's%d' % s2)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)

def r_link(graph, s, r, **attrs):
    nodes = ('s%d' % s, 'r%d' % r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


usage = """
Usage: python -m pysb.tools.render_reactions mymodel.py > mymodel.dot

Renders the reactions produced by a model into the "dot" graph format which can
be visualized with Graphviz.

To create a PDF from the .dot file, use the "dot" command from Graphviz:

    dot mymodel.dot -T pdf -O

This will create mymodel.dot.pdf. You can also change the "dot" command to one
of the other Graphviz drawing tools for a different type of layout. Note that
you can pipe the output of render_reactions straight into Graphviz without
creating an intermediate .dot file, which is especially helpful if you are
making continuous changes to the model and need to visualize your changes
repeatedly:

    python -m pysb.tools.render_species mymodel.py | dot -T pdf -o mymodel.pdf

Note that some PDF viewers will auto-reload a changed PDF, so you may not even
need to manually reopen it every time you rerun the tool.
"""
usage = usage[1:]  # strip leading newline

if __name__ == '__main__':
    # sanity checks on filename
    if len(sys.argv) <= 1:
        print usage,
        exit()
    model_filename = sys.argv[1]
    if not os.path.exists(model_filename):
        raise Exception("File '%s' doesn't exist" % model_filename)
    if not re.search(r'\.py$', model_filename):
        raise Exception("File '%s' is not a .py file" % model_filename)
    sys.path.insert(0, os.path.dirname(model_filename))
    model_name = re.sub(r'\.py$', '', os.path.basename(model_filename))
    # import it
    try:
        # FIXME if the model has the same name as some other "real" module
        # which we use, there will be trouble
        # (use the imp package and import as some safe name?)
        model_module = __import__(model_name)
    except StandardError as e:
        print "Error in model script:\n"
        raise
    # grab the 'model' variable from the module
    try:
        model = model_module.__dict__['model']
    except KeyError:
        raise Exception("File '%s' isn't a model file" % model_filename)
    print run(model).string()
