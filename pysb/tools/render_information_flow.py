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
import render_observables as ro

def networkx_itmprobe(raw_graph,name_attribute=True,model_name='emitting',use_weights=False,df=0.15):
    kwargs = {'df':df}
    if name_attribute:
        node_names = [raw_graph.node[n]['name'] for n in raw_graph.node]
    else:
        node_names = raw_graph.node.keys()

    # make sure all edge weights are positive by taking absolute value
    if use_weights:
        w = nx.get_edge_attributes(raw_graph,'weight')
	if len(w)==0:
            w = np.ones(len(raw_graph.edge))
        for wi in w:
            w[wi] = abs(w[wi])
        nx.set_edge_attributes(raw_graph,'weight',w)

    # convert to dense representation, as we need to set diagonals to non zero (use infinity)
    # this is so they don't get removed when converting to sparse representation, as
    # ITMProbe crashes when this happens
    graph = nx.to_numpy_matrix(raw_graph)
    di = np.diag_indices(len(node_names))
    graph[di] = float('inf')

    # now convert to sparse and supply to ITMProbe
    kwargs['G'] = CSRDirectedGraph(csr_matrix(graph), node_names)
    # convert the diagonal back to zero
    kwargs['G']._adjacency_matrix.data[kwargs['G']._diagonal_ix] = 0

    if not use_weights:
        # set non-zero values to unity, to make transition matrix uniform
        kwargs['G']._adjacency_matrix.data[kwargs['G']._adjacency_matrix.data!=0] = 1

    # set the model class
    model_class = model_classes[model_name]

    # assign memory for the result
    res = np.zeros(shape=(len(node_names),len(node_names)))

    # scan through the source nodes one at a time and assign to the result matrix
    for i,nm in enumerate(node_names):
        if model_name=='emitting':
            kwargs['source_nodes'] = [nm,]
            model = model_class(**kwargs)
            res[i,:] = model.H.T
        elif model_name=='absorbing':
            kwargs['sink_nodes'] = [nm,]
            model = model_class(**kwargs)
            res[i,:] = model.F.T

    return (res, node_names)

def run(model, include_monomers=True):
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
    obs_s_ids = []
    obs_species_mp = observable_monomer_patterns(model)
    if include_monomers:
    ## check where to find observable monomers in the list of species
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

    # for i, cp in enumerate(model.species):
    #     species_node = 's%d' % i
    #     slabel = re.sub(r'% ', r'%\\l', str(cp))
    #     slabel += '\\l'
    #     color = "#ccffcc"
    #     # color species with an initial condition differently
    #     if len([s for s in ic_species if s.is_equivalent_to(cp)]):
    #         color = "#aaffff"
    #     if i in all_obs_s_ids:
    #         color = "#ff66ff"
    #     graph.add_node(species_node,
    #                    label=slabel,
    #                    shape="Mrecord",
    #                    fillcolor=color, style="filled", color="transparent",
    #                    fontsize="12",
    #                    margin="0.06,0")


#     for i, reaction in enumerate(model.reactions_bidirectional):
# #        reaction_node = 'r%d' % i
# #        graph.add_node(reaction_node,
# #                       label=reaction_node,
# #                       shape="circle",
# #                       fillcolor="lightgray", style="filled", color="transparent",
# #                       fontsize="12",
# #                       width=".3", height=".3", margin="0.06,0")
#         reactants = set(reaction['reactants'])
#         products = set(reaction['products'])
#         modifiers = reactants & products
#         reactants = reactants - modifiers
#         products = products - modifiers
#         attr_reversible = {'dir': 'both', 'arrowtail': 'empty'} if reaction['reversible'] else {}
#         for s1 in reactants:
#             for s2 in products:
#                 s_link(graph, s1, s2, **attr_reversible)
#         for s1 in modifiers:
#             for s2 in modifiers:
#                 s_link(graph, s1, s2, arrowhead="odiamond")

    ## calculate information flow
    g = ro.run(model)
    raw_graph = nx.from_agraph(g)
    (res, node_names) = networkx_itmprobe(raw_graph, name_attribute=False)    

#    for i1,s1 in enumerate(node_names):
#        for i2,s2 in enumerate(node_names):
#            if res[i1,i2]>0:
#                s_link(graph,int(s1[1:]),int(s2[1:])) # remove the leading 's' from species IDs

    ## link the monomers
    if include_monomers:
        con1d = con2d = notcon = 0
        for oi1 in range(len(obs_species_mp)-1):
            for oi2 in range(oi1+1, len(obs_species_mp)):
                sp_ids1 = set(obs_s_ids[oi1])
                sp_ids2 = set(obs_s_ids[oi2])
                
                fwd_if = species_inf_flow(res, node_names, sp_ids1, sp_ids2)
                bkwd_if = species_inf_flow(res, node_names, sp_ids2, sp_ids1)
                # do they share a species?
                if len(set.intersection(sp_ids1,sp_ids2)) > 0:
                    m2m_link(graph, oi1, oi2, arrowtail='normal')
                    con2d = con2d + 1
                else:
                    if fwd_if and bkwd_if:
                        m2m_link(graph, oi1, oi2, arrowtail='normal')
                        con2d = con2d + 1
                    elif fwd_if:
                        m2m_link(graph, oi1, oi2)
                        con1d = con1d + 1
                    elif bkwd_if:
                        m2m_link(graph, oi2, oi1)
                        con1d = con1d + 1
                    else:
                        #print 'not connecting %d--%d' % (oi1,oi2)
                        notcon = notcon + 1

    print '%d monpat, %d con1d, %d con2d, %d not connected' % (len(obs_species_mp), con1d, con2d, notcon)
    return graph

def species_inf_flow(res, node_names, sp_ids1, sp_ids2):
    ''' search through two lists of species IDs for information flow '''
    for s1 in sp_ids1:
        for s2 in sp_ids2:
            if res[node_names.index('s%d' % s1),node_names.index('s%d' % s2)] > 0:
                return True
    return False

def m2m_link(graph, m1, m2, **attrs):
    nodes = ('m%d' %m1, 'm%d' %m2)
    attrs.setdefault('arrowhead','normal')
    graph.add_edge(*nodes, **attrs)

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
    run(model)
#    print run(model).string()
